const statusText = document.getElementById("status");
const preview = document.getElementById("preview");
const prediction = document.getElementById("prediction");
const heatmap = document.getElementById("heatmap");
const evaluationMetrics = document.getElementById("evaluation-metrics");
const downloadReportBtn = document.getElementById("download-report-btn");

let latestSubmission = null;

const setStatus = (message) => {
  if (statusText) {
    statusText.textContent = message;
  }
};

const dataUrl = sessionStorage.getItem("uploadedImageData");
const imageName = sessionStorage.getItem("uploadedImageName") || "uploaded-image.png";
const imageType = sessionStorage.getItem("uploadedImageType") || "image/png";
const deviceType = sessionStorage.getItem("uploadedDeviceType");
const shopQuoteAmount = (sessionStorage.getItem("shopQuoteAmount") || "").trim();
const shopQuoteCurrency = (sessionStorage.getItem("shopQuoteCurrency") || "").trim();
const shopQuoteDetails = (sessionStorage.getItem("shopQuoteDetails") || "").trim();
const regionCode = detectRegionCode();

loadEvaluationMetrics();

if (!dataUrl) {
  setStatus("No uploaded image found. Please upload an image first.");
} else if (!deviceType) {
  setStatus("Device type is missing. Please return and select phone or laptop.");
} else {
  preview.src = dataUrl;
  runPrediction(
    dataUrl,
    imageName,
    imageType,
    deviceType,
    regionCode,
    shopQuoteAmount,
    shopQuoteCurrency,
    shopQuoteDetails
  );
}

function detectRegionCode() {
  try {
    if (typeof Intl !== "undefined" && typeof Intl.Locale === "function") {
      const locale = new Intl.Locale(navigator.language || "en-US");
      if (locale.region) {
        return locale.region.toUpperCase();
      }
    }
  } catch (error) {
    // Ignore and use fallback parsing.
  }

  const lang = (navigator.language || "").toUpperCase();
  if (lang.includes("-")) {
    return lang.split("-")[1];
  }
  return "US";
}

function formatCurrency(amount, currency) {
  try {
    return new Intl.NumberFormat(navigator.language || "en-US", {
      style: "currency",
      currency,
      maximumFractionDigits: 2,
    }).format(amount);
  } catch (error) {
    return `${currency} ${Number(amount).toFixed(2)}`;
  }
}

function dataUrlToBlob(url) {
  const [meta, base64] = url.split(",");
  const mime = (meta.match(/data:(.*?);base64/) || [])[1] || "image/png";
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);

  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }

  return new Blob([bytes], { type: mime });
}

async function runPrediction(
  url,
  name,
  type,
  selectedDeviceType,
  selectedRegionCode,
  selectedQuoteAmount,
  selectedQuoteCurrency,
  selectedQuoteDetails
) {
  setStatus("Analyzing image...");
  prediction.innerHTML = "";
  heatmap.removeAttribute("src");

  const blob = dataUrlToBlob(url);
  const file = new File([blob], name, { type: type || blob.type || "image/png" });

  const formData = new FormData();
  formData.append("image", file);
  formData.append("device_type", selectedDeviceType);
  formData.append("region_code", selectedRegionCode);
  if (selectedQuoteAmount) {
    formData.append("shop_quote_amount", selectedQuoteAmount);
  }
  if (selectedQuoteCurrency) {
    formData.append("shop_quote_currency", selectedQuoteCurrency);
  }
  if (selectedQuoteDetails) {
    formData.append("shop_quote_details", selectedQuoteDetails);
  }

  latestSubmission = {
    url,
    name,
    type,
    selectedDeviceType,
    selectedRegionCode,
    selectedQuoteAmount,
    selectedQuoteCurrency,
    selectedQuoteDetails,
  };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let errorMessage = "Prediction failed";
      try {
        const errorJson = await response.json();
        errorMessage = errorJson.detail || errorMessage;
      } catch (error) {
        const errorText = await response.text();
        if (errorText) {
          errorMessage = errorText;
        }
      }
      throw new Error(errorMessage);
    }

    const data = await response.json();
    const localizedAmount = formatCurrency(data.cost_estimate.amount, data.cost_estimate.currency);
    const recommendation = data.repair_recommendation || {};
    const recommendationClass = recommendation.is_worth ? "recommendation worth" : "recommendation not-worth";
    const shopAssessment = data.shop_quote_assessment || null;
    const shopAssessmentClass =
      shopAssessment && shopAssessment.is_worth ? "recommendation worth" : "recommendation not-worth";
    const probs = Object.entries(data.severity.probabilities)
      .map(
        ([label, value]) =>
          `<li><strong>${label}:</strong> ${(value * 100).toFixed(1)}%</li>`
      )
      .join("");

    const quoteAmountMarkup =
      data.shop_quote && data.shop_quote.amount
        ? formatCurrency(data.shop_quote.amount, data.shop_quote.currency)
        : null;

    const shopAssessmentMarkup =
      shopAssessment && quoteAmountMarkup
        ? `
      <div class="${shopAssessmentClass}">
        <div class="decision">Shop Quote Decision: ${shopAssessment.decision}</div>
        <div class="message">Quoted Cost: ${quoteAmountMarkup}</div>
        ${data.shop_quote.details ? `<div class="reason"><strong>Shop Details:</strong> ${data.shop_quote.details}</div>` : ""}
        ${shopAssessment.reason ? `<div class="reason"><strong>Reason:</strong> ${shopAssessment.reason}</div>` : ""}
        ${shopAssessment.basis ? `<div class="basis">${shopAssessment.basis}</div>` : ""}
      </div>
    `
        : "";

    prediction.innerHTML = `
      <div class="prediction-main">
        <div class="confidence">Device: ${data.device_type.toUpperCase()}</div>
        <div class="severity">Severity: ${data.severity.label}</div>
        <div class="confidence">Confidence: ${(data.severity.confidence * 100).toFixed(1)}%</div>
      </div>
      <ul class="probability-list">${probs}</ul>
      <div class="cost">
        <div class="amount">Estimated Cost: ${localizedAmount}</div>
        <div class="note">${data.cost_estimate.note}</div>
      </div>
      <div class="${recommendationClass}">
        <div class="decision">${recommendation.decision || "Recommendation unavailable"}</div>
        <div class="message">${recommendation.message || "Try another image for a clearer recommendation."}</div>
        ${recommendation.reason ? `<div class="reason"><strong>Reason:</strong> ${recommendation.reason}</div>` : ""}
        ${recommendation.basis ? `<div class="basis">${recommendation.basis}</div>` : ""}
      </div>
      ${shopAssessmentMarkup}
    `;

    heatmap.src = `data:image/png;base64,${data.gradcam_base64}`;
    if (downloadReportBtn) {
      downloadReportBtn.disabled = false;
    }
    setStatus("Analysis complete.");
  } catch (error) {
    if (downloadReportBtn) {
      downloadReportBtn.disabled = true;
    }
    setStatus(`Error: ${error.message}`);
  }
}

function asPercent(value) {
  const num = Number(value || 0);
  const percentage = num >= 1 ? num : num * 100;
  return `${percentage.toFixed(1)}%`;
}

function clampPercent(value) {
  return Math.max(0, Math.min(100, Number(value || 0) * 100));
}

function renderEvaluationMetrics(payload) {
  if (!evaluationMetrics) {
    return;
  }

  const summary = payload.summary || {};
  const dataset = payload.dataset || {};
  const history = payload.training_history || {};
  const perClass = Array.isArray(payload.per_class) ? payload.per_class : [];
  const confusion = payload.confusion_matrix || {};
  const confusionLabels = Array.isArray(confusion.labels) ? confusion.labels : [];
  const confusionMatrix = Array.isArray(confusion.matrix) ? confusion.matrix : [];

  const summaryBars = [
    { label: "Accuracy", value: summary.accuracy || 0 },
    { label: "Macro Precision", value: summary.macro_precision || 0 },
    { label: "Macro Recall", value: summary.macro_recall || 0 },
    { label: "Macro F1", value: summary.macro_f1 || 0 },
  ]
    .map(
      (item) => `
        <div class="metric-row">
          <div class="metric-label">${item.label}</div>
          <div class="metric-track">
            <div class="metric-fill" style="width:${clampPercent(item.value)}%"></div>
          </div>
          <div class="metric-value">${asPercent(item.value)}</div>
        </div>
      `
    )
    .join("");

  const epochs = Math.max((history.accuracy || []).length, (history.val_accuracy || []).length);
  let epochBars = "";
  for (let i = 0; i < epochs; i += 1) {
    const trainAcc = Number((history.accuracy || [])[i] || 0);
    const valAcc = Number((history.val_accuracy || [])[i] || 0);
    epochBars += `
      <div class="epoch-row">
        <div class="epoch-label">E${i + 1}</div>
        <div class="epoch-bar-track">
          <div class="epoch-bar train" style="height:${Math.max(4, clampPercent(trainAcc))}%" title="Train: ${asPercent(trainAcc)}"></div>
          <div class="epoch-bar val" style="height:${Math.max(4, clampPercent(valAcc))}%" title="Validation: ${asPercent(valAcc)}"></div>
        </div>
      </div>
    `;
  }

  const perClassRows = perClass
    .map(
      (item) => `
        <tr>
          <td>${item.label}</td>
          <td>${asPercent(item.precision)}</td>
          <td>${asPercent(item.recall)}</td>
          <td>${asPercent(item.f1)}</td>
        </tr>
      `
    )
    .join("");

  const maxConfusionValue = confusionMatrix.length
    ? Math.max(...confusionMatrix.flat().map((value) => Number(value || 0)), 1)
    : 1;

  const confusionHeaderCells = confusionLabels
    .map((label) => `<th scope="col">${label}</th>`)
    .join("");

  const confusionRows = confusionMatrix
    .map((row, rowIdx) => {
      const label = confusionLabels[rowIdx] || `Class ${rowIdx + 1}`;
      const cells = row
        .map((value) => {
          const numeric = Number(value || 0);
          const intensity = Math.max(0.12, numeric / maxConfusionValue);
          return `<td class="cm-cell" style="--cm-intensity:${intensity}" title="${numeric}">${numeric}</td>`;
        })
        .join("");
      return `<tr><th scope="row">${label}</th>${cells}</tr>`;
    })
    .join("");

  evaluationMetrics.innerHTML = `
    <div class="metric-meta">
      <span>Model: ${payload.model_name || "N/A"}</span>
      <span>Train Samples: ${dataset.train_samples ?? "N/A"}</span>
      <span>Validation Samples: ${dataset.validation_samples ?? "N/A"}</span>
    </div>

    <div class="metric-chart">
      ${summaryBars}
    </div>

    <div class="epoch-chart">
      <div class="epoch-title">Epoch Accuracy (Train vs Validation)</div>
      <div class="epoch-bars">
        ${epochBars || '<div class="metric-empty">Training history not available.</div>'}
      </div>
      <div class="epoch-legend">
        <span><i class="legend-dot train"></i>Train</span>
        <span><i class="legend-dot val"></i>Validation</span>
      </div>
    </div>

    <table class="metrics-table">
      <thead>
        <tr>
          <th>Class</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
        </tr>
      </thead>
      <tbody>
        ${perClassRows || '<tr><td colspan="4">Per-class metrics not available.</td></tr>'}
      </tbody>
    </table>

    <div class="confusion-matrix-card">
      <div class="cm-title">Confusion Matrix (Actual x Predicted)</div>
      ${
        confusionLabels.length && confusionRows
          ? `
      <div class="cm-table-wrap">
        <table class="confusion-matrix-table">
          <thead>
            <tr>
              <th scope="col">Actual \ Predicted</th>
              ${confusionHeaderCells}
            </tr>
          </thead>
          <tbody>
            ${confusionRows}
          </tbody>
        </table>
      </div>`
          : '<div class="metric-empty">Confusion matrix is not available in this metrics file.</div>'
      }
    </div>
  `;
}

async function loadEvaluationMetrics() {
  if (!evaluationMetrics) {
    return;
  }

  evaluationMetrics.textContent = "Loading model metrics...";
  try {
    const response = await fetch(`/api/model/metrics?ts=${Date.now()}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      let detail = "";
      try {
        const errorPayload = await response.json();
        detail = errorPayload.detail || "";
      } catch (jsonError) {
        detail = "";
      }

      const baseMessage = detail || `Unable to load model metrics (HTTP ${response.status}).`;
      throw new Error(baseMessage);
    }

    const payload = await response.json();
    renderEvaluationMetrics(payload);
  } catch (error) {
    evaluationMetrics.innerHTML = `<div class="metric-empty">${error.message} If you just updated code, restart server and hard refresh this page.</div>`;
  }
}

if (downloadReportBtn) {
  downloadReportBtn.addEventListener("click", async () => {
    if (!latestSubmission) {
      setStatus("Run analysis first to download the PDF report.");
      return;
    }

    await downloadPdfReport(latestSubmission);
  });
}

async function downloadPdfReport(submission) {
  const {
    url,
    name,
    type,
    selectedDeviceType,
    selectedRegionCode,
    selectedQuoteAmount,
    selectedQuoteCurrency,
    selectedQuoteDetails,
  } = submission;

  try {
    if (downloadReportBtn) {
      downloadReportBtn.disabled = true;
      downloadReportBtn.textContent = "Preparing PDF...";
    }

    const blob = dataUrlToBlob(url);
    const file = new File([blob], name, { type: type || blob.type || "image/png" });
    const formData = new FormData();
    formData.append("image", file);
    formData.append("device_type", selectedDeviceType);
    formData.append("region_code", selectedRegionCode);

    if (selectedQuoteAmount) {
      formData.append("shop_quote_amount", selectedQuoteAmount);
    }
    if (selectedQuoteCurrency) {
      formData.append("shop_quote_currency", selectedQuoteCurrency);
    }
    if (selectedQuoteDetails) {
      formData.append("shop_quote_details", selectedQuoteDetails);
    }

    const response = await fetch("/api/predict/report", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || "Unable to generate PDF report.");
    }

    const pdfBlob = await response.blob();
    const link = document.createElement("a");
    const objectUrl = URL.createObjectURL(pdfBlob);
    link.href = objectUrl;
    link.download = "damage-prediction-report.pdf";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
    setStatus("PDF report downloaded.");
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  } finally {
    if (downloadReportBtn) {
      downloadReportBtn.disabled = false;
      downloadReportBtn.textContent = "Download PDF Report";
    }
  }
}
