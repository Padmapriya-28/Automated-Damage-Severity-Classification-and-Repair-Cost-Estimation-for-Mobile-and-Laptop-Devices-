const statusText = document.getElementById("status");
const preview = document.getElementById("preview");
const prediction = document.getElementById("prediction");
const heatmap = document.getElementById("heatmap");
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
