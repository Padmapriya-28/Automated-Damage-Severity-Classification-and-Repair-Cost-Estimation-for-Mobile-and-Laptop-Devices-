import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from data_preprocessing.preprocess import preprocess_image
from models.classification_model import get_classifier
from models.cost_estimation_model import get_cost_estimator
from utils.currency import convert_from_usd, convert_to_usd, resolve_currency
from utils.device_validator import validate_device_image
from utils.gradcam import generate_gradcam_overlay

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)
WEBSITE_NAME = "Damage Severity & Repair Cost Intelligence"

REPLACEMENT_BENCHMARK_USD = {
    "phone": 450.0,
    "laptop": 900.0,
}


@api_bp.post("/predict")
def predict_damage():
    payload_or_error, status = build_prediction_payload(request)
    if status != 200:
        return jsonify(payload_or_error), status
    return jsonify(payload_or_error), 200


@api_bp.get("/model/metrics")
def get_model_metrics():
    weights_dir = Path(__file__).resolve().parents[1] / "models" / "weights"
    metrics_path = weights_dir / "evaluation_metrics.json"
    if not metrics_path.exists():
        return (
            jsonify(
                {
                    "detail": "Model evaluation metrics are not available yet. Run training to generate them.",
                    "metrics_path": str(metrics_path),
                }
            ),
            404,
        )

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        logger.exception("Failed to read evaluation metrics file")
        return jsonify({"detail": "Failed to load model evaluation metrics."}), 500

    return jsonify(metrics), 200


@api_bp.post("/predict/report")
def predict_damage_report():
    payload_or_error, status = build_prediction_payload(request)
    if status != 200:
        return jsonify(payload_or_error), status

    pdf_bytes = build_pdf_report(payload_or_error)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="damage-prediction-report.pdf",
    )


def build_prediction_payload(req):
    image = req.files.get("image")
    device_type = (req.form.get("device_type") or "").strip().lower()
    region_code = (req.form.get("region_code") or "").strip().upper()
    currency_code = (req.form.get("currency_code") or "").strip().upper()
    shop_quote_amount_raw = (req.form.get("shop_quote_amount") or "").strip()
    shop_quote_currency_raw = (req.form.get("shop_quote_currency") or "").strip().upper()
    shop_quote_details = (req.form.get("shop_quote_details") or "").strip()

    if image is None:
        return {"detail": "Image file is required."}, 400

    normalized_device = device_type.strip().lower()
    if normalized_device not in {"phone", "laptop"}:
        return {"detail": "device_type must be either 'phone' or 'laptop'."}, 400

    content_type = (image.mimetype or "").lower()
    if content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        return {"detail": "Only JPG and PNG images are supported."}, 400

    try:
        image_bytes = image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        logger.exception("Failed to read image")
        return {"detail": "Invalid image file."}, 400

    is_valid_device, validation_reason, _ = validate_device_image(pil_image, normalized_device)
    if not is_valid_device:
        return {"detail": validation_reason}, 400

    classifier = get_classifier()
    estimator = get_cost_estimator()

    processed, display_image = preprocess_image(pil_image, classifier.model_name)
    severity_label, confidence, probs = classifier.predict(processed)

    try:
        heatmap = generate_gradcam_overlay(
            classifier.model,
            processed,
            display_image,
            classifier.last_conv_layer,
            classifier.label_to_index[severity_label],
        )
    except Exception:
        logger.exception("Grad-CAM generation failed. Falling back to original image.")
        heatmap = display_image

    cost_amount_usd, cost_note = estimator.estimate(
        severity_label,
        normalized_device,
        confidence,
        probs,
    )
    resolved_currency = resolve_currency(region_code, currency_code)
    cost_amount_local = convert_from_usd(cost_amount_usd, resolved_currency)

    shop_quote = None
    shop_quote_assessment = None
    if shop_quote_amount_raw:
        try:
            shop_quote_amount_local = float(shop_quote_amount_raw)
        except ValueError:
            return {"detail": "shop_quote_amount must be a valid number."}, 400

        if shop_quote_amount_local <= 0:
            return {"detail": "shop_quote_amount must be greater than 0."}, 400

        quote_currency = resolve_currency(region_code, shop_quote_currency_raw)
        shop_quote_amount_usd = convert_to_usd(shop_quote_amount_local, quote_currency)
        shop_quote = {
            "amount": round(shop_quote_amount_local, 2),
            "currency": quote_currency,
            "amount_usd": shop_quote_amount_usd,
            "details": shop_quote_details,
        }
        shop_quote_assessment = build_shop_quote_assessment(
            device_type=normalized_device,
            severity_label=severity_label,
            estimated_cost_usd=cost_amount_usd,
            quoted_cost_usd=shop_quote_amount_usd,
        )

    recommendation = build_repair_recommendation(
        device_type=normalized_device,
        severity_label=severity_label,
        estimated_cost_usd=cost_amount_usd,
    )

    buffered = io.BytesIO()
    heatmap.save(buffered, format="PNG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = {
        "device_type": normalized_device,
        "region_code": region_code or "AUTO",
        "severity": {
            "label": severity_label,
            "confidence": confidence,
            "probabilities": probs,
        },
        "cost_estimate": {
            "amount": cost_amount_local,
            "currency": resolved_currency,
            "amount_usd": cost_amount_usd,
            "note": f"{cost_note}; converted from USD using regional currency",
        },
        "repair_recommendation": recommendation,
        "shop_quote": shop_quote,
        "shop_quote_assessment": shop_quote_assessment,
        "gradcam_base64": heatmap_base64,
        "model_name": classifier.model_name,
    }
    return response, 200


def build_pdf_report(payload: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    severity = payload.get("severity", {})
    cost = payload.get("cost_estimate", {})
    recommendation = payload.get("repair_recommendation", {})
    shop_quote = payload.get("shop_quote")
    shop_assessment = payload.get("shop_quote_assessment")
    probabilities = severity.get("probabilities", {})

    story.append(Paragraph(f"<b>{WEBSITE_NAME}</b>", styles["Title"]))
    story.append(Paragraph("Damage Device Prediction Report", styles["Heading2"]))
    story.append(Paragraph(datetime.utcnow().strftime("Generated on %Y-%m-%d %H:%M UTC"), styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Device Type:</b> {str(payload.get('device_type', 'N/A')).upper()}", styles["Normal"]))
    story.append(Paragraph(f"<b>Region:</b> {payload.get('region_code', 'AUTO')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Severity:</b> {severity.get('label', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Confidence:</b> {round(float(severity.get('confidence', 0)) * 100, 1)}%", styles["Normal"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Probability Breakdown</b>", styles["Heading3"]))
    if probabilities:
        for label, value in probabilities.items():
            story.append(Paragraph(f"- {label}: {round(float(value) * 100, 1)}%", styles["Normal"]))
    else:
        story.append(Paragraph("- Not available", styles["Normal"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Estimated Repair Cost</b>", styles["Heading3"]))
    story.append(Paragraph(f"<b>Local:</b> {cost.get('currency', 'USD')} {cost.get('amount', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"<b>USD Baseline:</b> USD {cost.get('amount_usd', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Cost Note:</b> {cost.get('note', 'N/A')}", styles["Normal"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>AI Worthiness Recommendation</b>", styles["Heading3"]))
    story.append(Paragraph(f"<b>Decision:</b> {recommendation.get('decision', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Reason:</b> {recommendation.get('reason', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Basis:</b> {recommendation.get('basis', 'N/A')}", styles["Normal"]))

    if shop_quote:
        story.append(Spacer(1, 8))
        story.append(Paragraph("<b>Shop Quote Information</b>", styles["Heading3"]))
        story.append(Paragraph(f"<b>Quoted Amount:</b> {shop_quote.get('currency', 'USD')} {shop_quote.get('amount', 'N/A')}", styles["Normal"]))
        story.append(Paragraph(f"<b>Quoted Amount (USD):</b> USD {shop_quote.get('amount_usd', 'N/A')}", styles["Normal"]))
        story.append(Paragraph(f"<b>Shop Details:</b> {shop_quote.get('details') or 'N/A'}", styles["Normal"]))

    if shop_assessment:
        story.append(Spacer(1, 8))
        story.append(Paragraph("<b>Shop Quote Decision</b>", styles["Heading3"]))
        story.append(Paragraph(f"<b>Decision:</b> {shop_assessment.get('decision', 'N/A')}", styles["Normal"]))
        story.append(Paragraph(f"<b>Reason:</b> {shop_assessment.get('reason', 'N/A')}", styles["Normal"]))
        story.append(Paragraph(f"<b>Basis:</b> {shop_assessment.get('basis', 'N/A')}", styles["Normal"]))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def build_repair_recommendation(device_type: str, severity_label: str, estimated_cost_usd: float) -> dict:
    benchmark = REPLACEMENT_BENCHMARK_USD.get(device_type, 600.0)
    ratio = estimated_cost_usd / benchmark if benchmark > 0 else 1.0

    if ratio <= 0.45:
        return {
            "is_worth": True,
            "decision": "Worth Repairing",
            "message": "Repair is financially reasonable compared to replacement cost.",
            "reason": "Estimated repair cost is low relative to typical replacement value.",
            "cost_ratio": round(ratio, 2),
            "basis": f"Estimated repair is about {round(ratio * 100)}% of a typical {device_type} replacement value.",
        }

    if ratio <= 0.7 and severity_label in {"Minor", "Moderate"}:
        return {
            "is_worth": True,
            "decision": "Conditionally Worth Repairing",
            "message": "Repair may be worthwhile if the device is otherwise in good condition.",
            "reason": "Repair cost is moderate and damage severity is not at the highest level.",
            "cost_ratio": round(ratio, 2),
            "basis": f"Estimated repair is about {round(ratio * 100)}% of a typical {device_type} replacement value.",
        }

    return {
        "is_worth": False,
        "decision": "Likely Not Worth Repairing",
        "message": "Replacement is often more cost-effective than repair for this damage level.",
        "reason": "Estimated repair cost is high relative to replacement value or severity is too high.",
        "cost_ratio": round(ratio, 2),
        "basis": f"Estimated repair is about {round(ratio * 100)}% of a typical {device_type} replacement value.",
    }


def build_shop_quote_assessment(
    device_type: str,
    severity_label: str,
    estimated_cost_usd: float,
    quoted_cost_usd: float,
) -> dict:
    benchmark = REPLACEMENT_BENCHMARK_USD.get(device_type, 600.0)
    quoted_ratio = quoted_cost_usd / benchmark if benchmark > 0 else 1.0
    model_diff = quoted_cost_usd - estimated_cost_usd
    model_diff_pct = (model_diff / estimated_cost_usd) if estimated_cost_usd > 0 else 0.0

    if quoted_ratio <= 0.55 and model_diff_pct <= 0.2:
        return {
            "is_worth": True,
            "decision": "Yes, Worth Repairing",
            "reason": "Shop quote is close to model estimate and still well below replacement value.",
            "basis": f"Quote is {round(quoted_ratio * 100)}% of replacement value and {round(model_diff_pct * 100)}% vs model estimate.",
        }

    if quoted_ratio <= 0.72 and severity_label in {"Minor", "Moderate"}:
        return {
            "is_worth": True,
            "decision": "Yes, but Review Carefully",
            "reason": "Quote is moderate; repair can be justified if device condition and warranty are acceptable.",
            "basis": f"Quote is {round(quoted_ratio * 100)}% of replacement value and {round(model_diff_pct * 100)}% vs model estimate.",
        }

    return {
        "is_worth": False,
        "decision": "No, Likely Not Worth Repairing",
        "reason": "Quoted repair cost is high compared to replacement value or above expected estimate.",
        "basis": f"Quote is {round(quoted_ratio * 100)}% of replacement value and {round(model_diff_pct * 100)}% vs model estimate.",
    }
