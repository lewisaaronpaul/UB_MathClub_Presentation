const previewContainer = document.getElementById("imagePreview");
const picFile = document.getElementById("picFile");
const previewImage = previewContainer.querySelector(".image-preview__image");
const previewDefaultText = previewContainer.querySelector(".image-preview__default-text");

picFile.addEventListener("change", function () {
  const theFile = this.files[0];

  if (theFile) {
    const reader = new FileReader();

    previewDefaultText.style.display = "none";
    previewImage.style.display = "block";

    reader.addEventListener("load", function () {
      previewImage.setAttribute("src", this.result);
    });

    reader.readAsDataURL(theFile);
  } else {
    previewDefaultText.style.display = null;
    previewImage.style.display = null;
    previewImage.setAttribute("src", "");
  }
}); 