function enviarImagen() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert('Por favor, selecciona una imagen primero.');
        return;
    }

    const reader = new FileReader();
    reader.onloadend = function () {
        // Convertir la imagen a Base64
        const base64Image = reader.result.split(',')[1];  // Eliminar la parte "data:image/jpeg;base64,"
        
        // Crear la solicitud HTTP POST
        const formData = new FormData();
        formData.append("data", base64Image);

        fetch("http://localhost:5000/predict", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Mostrar los resultados
            document.getElementById('prediccion').textContent = `Predicción: ${data.prediccion}`;
            document.getElementById('confianza').textContent = `Confianza: ${data.confianza}`;
            document.getElementById('resultado').style.display = 'block';
        })
        .catch(error => {
            console.error("Error en la solicitud:", error);
            alert('Error al enviar la imagen');
        });
    };
    
    reader.readAsDataURL(file);  // Leer la imagen como Data URL
}
