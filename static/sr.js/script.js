document.getElementById('planForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const userInput = {
        bedrooms: document.getElementById('bedrooms').value,
        bathrooms: document.getElementById('bathrooms').value,
        kitchen: document.getElementById('kitchen').value,
        living_room: document.getElementById('living_room').value,
        garage: document.getElementById('garage').value,
        sqft: document.getElementById('sqft').value
    };

    const response = await fetch('/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(userInput)
    });

    const result = await response.json();
    document.getElementById('floorPlanImage').src = result.floorPlanImage;  // Modify to show result in the desired format
});
