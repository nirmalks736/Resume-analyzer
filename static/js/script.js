document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    const fileInput = document.querySelector("input[type='file']");
    const uploadButton = document.querySelector("button");

    form.addEventListener("submit", function () {
        if (fileInput.files.length === 0) {
            alert("Please select a PDF file to upload.");
            return false;
        }

        uploadButton.innerText = "Uploading...";
        uploadButton.disabled = true;
    });

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            const fileName = fileInput.files[0].name;
            alert('Selected file: ${fileName}');
        }
    });
});