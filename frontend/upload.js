const form = document.getElementById("upload-form");
const takePhotoBtn = document.getElementById("take-photo-btn");
const uploadPhotoBtn = document.getElementById("upload-photo-btn");
const cameraInput = document.getElementById("camera-input");
const galleryInput = document.getElementById("gallery-input");
const deviceTypeInput = document.getElementById("device-type");
const shopQuoteAmountInput = document.getElementById("shop-quote-amount");
const shopQuoteCurrencyInput = document.getElementById("shop-quote-currency");
const shopQuoteDetailsInput = document.getElementById("shop-quote-details");
const statusText = document.getElementById("status");

let selectedImageFile = null;

const setStatus = (message) => {
  statusText.textContent = message;
};

takePhotoBtn.addEventListener("click", () => {
  cameraInput.click();
});

uploadPhotoBtn.addEventListener("click", () => {
  galleryInput.click();
});

cameraInput.addEventListener("change", () => {
  const file = cameraInput.files[0];
  updateSelectedImage(file, "camera");
});

galleryInput.addEventListener("change", () => {
  const file = galleryInput.files[0];
  updateSelectedImage(file, "gallery");
});

function updateSelectedImage(file, source) {
  if (!file) {
    selectedImageFile = null;
    setStatus("");
    return;
  }

  selectedImageFile = file;
  const sourceLabel = source === "camera" ? "camera" : "gallery";
  setStatus(`Image selected from ${sourceLabel}. Continue to prediction.`);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const file = selectedImageFile;
  const deviceType = deviceTypeInput.value;
  const shopQuoteAmount = (shopQuoteAmountInput.value || "").trim();
  const shopQuoteCurrency = (shopQuoteCurrencyInput.value || "").trim();
  const shopQuoteDetails = (shopQuoteDetailsInput.value || "").trim();

  if (!file) {
    setStatus("Please select an image.");
    return;
  }

  if (!deviceType) {
    setStatus("Please select the device type (phone or laptop).");
    return;
  }

  if (shopQuoteAmount) {
    const parsedAmount = Number(shopQuoteAmount);
    if (!Number.isFinite(parsedAmount) || parsedAmount <= 0) {
      setStatus("Please enter a valid shop quote amount greater than 0.");
      return;
    }
  }

  const reader = new FileReader();
  reader.onload = () => {
    sessionStorage.setItem("uploadedImageData", reader.result);
    sessionStorage.setItem("uploadedImageName", file.name);
    sessionStorage.setItem("uploadedImageType", file.type || "image/png");
    sessionStorage.setItem("uploadedDeviceType", deviceType);
    sessionStorage.setItem("shopQuoteAmount", shopQuoteAmount);
    sessionStorage.setItem("shopQuoteCurrency", shopQuoteCurrency);
    sessionStorage.setItem("shopQuoteDetails", shopQuoteDetails);
    window.location.href = "/static/prediction.html";
  };

  reader.onerror = () => {
    setStatus("Unable to read image file. Please try another one.");
  };

  setStatus("Redirecting to prediction page...");
  reader.readAsDataURL(file);
});
