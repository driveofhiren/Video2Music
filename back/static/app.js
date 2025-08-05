document.addEventListener('DOMContentLoaded', () => {
	const uploadForm = document.getElementById('uploadForm')
	const liveBtn = document.getElementById('liveBtn')
	const statusDiv = document.getElementById('status')

	uploadForm.addEventListener('submit', async (e) => {
		e.preventDefault()
		const fileInput = document.getElementById('mediaInput')
		if (!fileInput.files.length) return

		statusDiv.textContent = 'Uploading...'

		const formData = new FormData()
		for (const file of fileInput.files) {
			formData.append('files', file)
		}

		try {
			const response = await fetch('/upload', {
				method: 'POST',
				body: formData,
			})
			const result = await response.json()

			if (result.status === 'success') {
				statusDiv.textContent = 'Processing complete! Music is playing.'
			} else {
				statusDiv.textContent = `Error: ${result.message}`
			}
		} catch (error) {
			statusDiv.textContent = `Upload failed: ${error.message}`
		}
	})

	liveBtn.addEventListener('click', async () => {
		statusDiv.textContent = 'Connecting to live camera...'
		const socket = new WebSocket(`ws://${window.location.host}/ws`)

		socket.onopen = () => {
			statusDiv.textContent = 'Live processing started!'
		}

		socket.onmessage = (event) => {
			const data = JSON.parse(event.data)
			if (data.error) {
				statusDiv.textContent = `Error: ${data.error}`
			} else if (data.status) {
				statusDiv.textContent = data.status
			}
		}

		socket.onclose = () => {
			statusDiv.textContent = 'Live session ended'
		}
	})
})
