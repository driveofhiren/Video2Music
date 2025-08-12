class LiveMusicGenerator {
	constructor() {
		this.initializeElements()
		this.initializeVariables()
		this.setupEventListeners()
		this.initializeApplication()
	}

	initializeElements() {
		// DOM Elements
		this.audioPlayer = document.getElementById('audioPlayer')
		this.connectionStatus = document.getElementById('connectionStatus')
		this.cameraStatus = document.getElementById('cameraStatus')
		this.liveStatus = document.getElementById('liveStatus')
		this.liveBtn = document.getElementById('liveBtn')
		this.startCameraBtn = document.getElementById('startCameraBtn')
		this.stopMusicBtn = document.getElementById('stopMusicBtn')
		this.audioVisualizer = document.getElementById('audioVisualizer')
		this.videoPreview = document.getElementById('videoPreview')
		this.videoPlaceholder = document.getElementById('videoPlaceholder')
		this.captureCanvas = document.getElementById('captureCanvas')
		this.bpmSlider = document.getElementById('bpmSlider')
		this.bpmValue = document.getElementById('bpmValue')
		this.scaleSelect = document.getElementById('scaleSelect')
		this.genreInput = document.getElementById('genreInput')

		// Status indicators
		this.cameraStatusDot = document.getElementById('cameraStatusDot')
		this.musicStatusDot = document.getElementById('musicStatusDot')
		this.audioStatusDot = document.getElementById('audioStatusDot')
	}

	initializeVariables() {
		// Audio context and analysis
		this.audioContext = null
		this.analyser = null
		this.audioSource = null
		this.visualizationInterval = null

		// WebSocket connections
		this.audioWs = null
		this.videoWs = null

		// Media stream
		this.mediaStream = null
		this.captureContext = null
		this.frameInterval = null

		// Connection management
		this.reconnectAttempts = 0
		this.maxReconnectAttempts = 5
		this.keepAliveInterval = null

		// Configuration
		this.sampleRate = 48000
		this.frameRate = 10 // FPS for video streaming
	}

	setupEventListeners() {
		this.startCameraBtn.addEventListener('click', () =>
			this.initializeCamera()
		)

		// BPM slider with real-time value update
		this.bpmSlider.addEventListener('input', (e) => {
			this.bpmValue.textContent = e.target.value
			this.animateBpmChange()
		})

		this.liveBtn.addEventListener('click', () => this.startLiveProcessing())
		this.stopMusicBtn.addEventListener('click', () =>
			this.stopMusicGeneration()
		)

		// Page visibility handling
		document.addEventListener('visibilitychange', () => {
			if (document.visibilityState === 'visible') {
				console.log('Page became visible, reconnecting audio...')
				this.connectAudioWebSocket()
			} else {
				console.log('Page hidden, cleaning up...')
				this.stopKeepAlive()
			}
		})

		// Cleanup on page unload
		window.addEventListener('beforeunload', () => this.cleanup())
	}

	animateBpmChange() {
		const bpmDisplay = this.bpmValue.parentElement
		bpmDisplay.style.transform = 'scale(1.1)'
		bpmDisplay.style.color = 'var(--accent)'
		setTimeout(() => {
			bpmDisplay.style.transform = 'scale(1)'
			bpmDisplay.style.color = 'var(--primary)'
		}, 200)
	}

	async initializeApplication() {
		// Check for required APIs
		if (!this.checkBrowserCompatibility()) {
			return
		}

		// Initialize audio context
		this.initAudioContext()

		// Connect to audio WebSocket
		this.connectAudioWebSocket()
	}

	checkBrowserCompatibility() {
		if (!window.WebSocket) {
			this.updateStatus(
				this.connectionStatus,
				'WebSocket not supported in this browser',
				'error'
			)
			return false
		}

		if (!(window.AudioContext || window.webkitAudioContext)) {
			this.updateStatus(
				this.connectionStatus,
				'Web Audio API not supported',
				'error'
			)
			return false
		}

		if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
			this.updateStatus(
				this.cameraStatus,
				'Camera access not supported in this browser',
				'error'
			)
			this.startCameraBtn.disabled = true
			return false
		}

		return true
	}

	// Audio Context Management
	initAudioContext() {
		try {
			if (this.audioContext && this.audioContext.state !== 'closed') {
				return this.audioContext
			}

			this.audioContext = new (window.AudioContext ||
				window.webkitAudioContext)({
				sampleRate: this.sampleRate,
			})

			this.analyser = this.audioContext.createAnalyser()
			this.analyser.fftSize = 512

			const gainNode = this.audioContext.createGain()
			gainNode.gain.value = 1.0

			this.analyser.connect(gainNode)
			gainNode.connect(this.audioContext.destination)

			this.setupVisualization()

			console.log(
				'Audio context initialized with sample rate:',
				this.audioContext.sampleRate
			)
			return this.audioContext
		} catch (e) {
			console.error('Error initializing audio context:', e)
			this.updateStatus(
				this.connectionStatus,
				'Error initializing audio',
				'error'
			)
			return null
		}
	}

	setupVisualization() {
		if (this.visualizationInterval) {
			clearInterval(this.visualizationInterval)
		}

		const canvas = document.createElement('canvas')
		const rect = this.audioVisualizer.getBoundingClientRect()
		canvas.width = rect.width * window.devicePixelRatio || rect.width
		canvas.height = rect.height * window.devicePixelRatio || rect.height
		canvas.style.width = rect.width + 'px'
		canvas.style.height = rect.height + 'px'

		this.audioVisualizer.innerHTML = ''
		this.audioVisualizer.appendChild(canvas)

		const ctx = canvas.getContext('2d')
		ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1)

		// Enhanced gradient
		const gradient = ctx.createLinearGradient(0, 0, 0, rect.height)
		gradient.addColorStop(0, 'rgba(6, 182, 212, 0.8)')
		gradient.addColorStop(0.3, 'rgba(99, 102, 241, 0.6)')
		gradient.addColorStop(0.7, 'rgba(139, 92, 246, 0.4)')
		gradient.addColorStop(1, 'rgba(6, 182, 212, 0.2)')

		this.visualizationInterval = setInterval(() => {
			if (!this.analyser) return

			const bufferLength = this.analyser.frequencyBinCount
			const dataArray = new Uint8Array(bufferLength)
			this.analyser.getByteFrequencyData(dataArray)

			// Clear with smooth fade
			ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'
			ctx.fillRect(0, 0, rect.width, rect.height)

			// Enhanced visualization
			const barWidth = (rect.width / bufferLength) * 2.5
			let x = 0

			for (let i = 0; i < bufferLength; i++) {
				const barHeight = (dataArray[i] / 255) * rect.height * 0.9

				// Add glow effect
				ctx.shadowColor = '#06b6d4'
				ctx.shadowBlur = 10
				ctx.fillStyle = gradient

				// Rounded bars
				ctx.beginPath()
				ctx.roundRect(
					x,
					rect.height - barHeight,
					barWidth - 2,
					barHeight,
					4
				)
				ctx.fill()

				x += barWidth + 1
			}

			ctx.shadowBlur = 0
		}, 30) // Smoother animation at ~33fps
	}

	// Camera Management
	async initializeCamera() {
		try {
			this.updateStatus(
				this.cameraStatus,
				'Initializing camera system...',
				'processing'
			)

			// Stop existing stream if any
			if (this.mediaStream) {
				this.mediaStream.getTracks().forEach((track) => track.stop())
			}

			// Request camera access with enhanced settings
			this.mediaStream = await navigator.mediaDevices.getUserMedia({
				video: {
					width: { ideal: 1280, min: 640 },
					height: { ideal: 720, min: 480 },
					frameRate: { ideal: 30, min: 15 },
					facingMode: { ideal: 'environment' },
				},
			})

			// Setup video element
			this.videoPreview.srcObject = this.mediaStream
			this.videoPreview.style.display = 'block'
			this.videoPlaceholder.style.display = 'none'

			// Setup canvas for frame capture
			this.captureCanvas.width = 640
			this.captureCanvas.height = 480
			this.captureContext = this.captureCanvas.getContext('2d')

			// Update UI
			this.startCameraBtn.disabled = true
			this.liveBtn.disabled = false

			this.updateStatus(
				this.cameraStatus,
				'Camera system ready',
				'connected'
			)

			console.log('Camera initialized successfully')
		} catch (error) {
			console.error('Camera initialization failed:', error)
			this.updateStatus(
				this.cameraStatus,
				`Camera access failed: ${error.message}`,
				'error'
			)

			// Reset UI on error
			this.startCameraBtn.disabled = false
			this.liveBtn.disabled = true
		}
	}

	// WebSocket Management
	connectVideoWebSocket() {
		const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
		const host = window.location.host
		this.videoWs = new WebSocket(`${protocol}//${host}/ws/video`)

		this.videoWs.onopen = () => {
			console.log('Video WebSocket connected')
			this.startVideoStreaming()
		}

		this.videoWs.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data)
				if (data.error) {
					console.error('Server error:', data.error)
				}
			} catch (e) {
				console.log('Non-JSON message:', event.data)
			}
		}

		this.videoWs.onclose = (event) => {
			console.log('Video WebSocket closed:', event.code)
			if (this.frameInterval) {
				clearInterval(this.frameInterval)
				this.frameInterval = null
			}
		}

		this.videoWs.onerror = (error) => {
			console.error('Video WebSocket error:', error)
		}
	}

	startVideoStreaming() {
		if (this.frameInterval) {
			clearInterval(this.frameInterval)
		}

		this.frameInterval = setInterval(() => {
			if (
				this.videoPreview &&
				this.captureContext &&
				this.videoWs &&
				this.videoWs.readyState === WebSocket.OPEN
			) {
				try {
					// Draw current video frame to canvas
					this.captureContext.drawImage(
						this.videoPreview,
						0,
						0,
						640,
						480
					)

					// Convert to base64 JPEG
					const frameData = this.captureCanvas.toDataURL(
						'image/jpeg',
						0.8
					)

					// Send frame to server
					this.videoWs.send(
						JSON.stringify({
							frame: frameData,
							timestamp: Date.now(),
						})
					)
				} catch (error) {
					console.error('Error capturing frame:', error)
				}
			}
		}, 1000 / this.frameRate)
	}

	connectAudioWebSocket() {
		if (this.audioWs) {
			try {
				this.audioWs.close()
			} catch (e) {
				console.log('Error closing previous audio connection:', e)
			}
		}

		this.updateStatus(
			this.connectionStatus,
			'Establishing audio connection...',
			'processing'
		)

		const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
		const host = window.location.host
		this.audioWs = new WebSocket(`${protocol}//${host}/ws/audio`)

		this.audioWs.binaryType = 'arraybuffer'

		this.audioWs.onopen = () => {
			this.reconnectAttempts = 0
			this.updateStatus(
				this.connectionStatus,
				'Audio stream connected',
				'connected'
			)
			this.initAudioContext()
			this.startKeepAlive()
			console.log('Audio WebSocket connection established')
		}

		this.audioWs.onmessage = async (event) => {
			if (event.data instanceof ArrayBuffer) {
				await this.processAudioData(event.data)
			} else if (typeof event.data === 'string') {
				if (event.data === 'pong') {
					console.log('Audio keepalive received')
				} else {
					console.log('Server message:', event.data)
				}
			}
		}

		this.audioWs.onclose = (event) => {
			this.stopKeepAlive()
			this.handleAudioDisconnection(event)
		}

		this.audioWs.onerror = (error) => {
			console.error('Audio WebSocket error:', error)
			this.updateStatus(
				this.connectionStatus,
				'Audio connection error',
				'error'
			)
		}
	}

	async processAudioData(arrayBuffer) {
		try {
			const audioContext = this.initAudioContext()
			if (!audioContext) return

			// Try decoding as WAV first
			try {
				const audioData = await audioContext.decodeAudioData(
					arrayBuffer
				)
				this.playAudioData(audioData)
				return
			} catch (wavError) {
				console.log('WAV decode failed, trying raw PCM:', wavError)
			}

			// If WAV fails, try raw PCM
			try {
				const audioData = this.decodeRawPCM(arrayBuffer, audioContext)
				this.playAudioData(audioData)
			} catch (pcmError) {
				console.error('PCM decode failed:', pcmError)
				this.updateStatus(
					this.connectionStatus,
					'Error decoding audio',
					'error'
				)
			}
		} catch (e) {
			console.error('Error processing audio:', e)
			this.updateStatus(
				this.connectionStatus,
				'Error processing audio',
				'error'
			)
		}
	}

	handleAudioDisconnection(event) {
		if (event.code === 1000) {
			this.updateStatus(
				this.connectionStatus,
				'Audio connection closed',
				'processing'
			)
		} else {
			const message =
				event.code === 1006
					? 'Audio connection failed'
					: 'Audio connection closed'
			this.updateStatus(
				this.connectionStatus,
				`${message} (code ${event.code}) - reconnecting...`,
				'error'
			)

			if (this.reconnectAttempts < this.maxReconnectAttempts) {
				this.reconnectAttempts++
				const delay = Math.min(1000 * this.reconnectAttempts, 5000)
				setTimeout(() => this.connectAudioWebSocket(), delay)
			} else {
				this.updateStatus(
					this.connectionStatus,
					'Max audio reconnection attempts reached',
					'error'
				)
			}
		}
	}

	// Audio Processing
	playAudioData(audioData) {
		if (!this.audioContext) {
			this.initAudioContext()
			if (!this.audioContext) return
		}

		// Stop previous source if exists
		if (this.audioSource) {
			try {
				this.audioSource.stop()
			} catch (e) {
				console.log('Error stopping previous source:', e)
			}
		}

		// Create new source
		this.audioSource = this.audioContext.createBufferSource()
		this.audioSource.buffer = audioData
		this.audioSource.connect(this.analyser)
		this.audioSource.start()

		// Visual feedback
		this.updateStatus(
			this.connectionStatus,
			'Streaming live music...',
			'connected'
		)
	}

	decodeRawPCM(arrayBuffer, audioContext) {
		const numChannels = 2 // Stereo
		const bytesPerSample = 2 // 16-bit
		const totalSamples =
			arrayBuffer.byteLength / (numChannels * bytesPerSample)

		const audioBuffer = audioContext.createBuffer(
			numChannels,
			totalSamples,
			audioContext.sampleRate
		)

		const view = new DataView(arrayBuffer)

		for (let channel = 0; channel < numChannels; channel++) {
			const channelData = audioBuffer.getChannelData(channel)
			for (let i = 0; i < totalSamples; i++) {
				const offset = (i * numChannels + channel) * bytesPerSample
				const sample = view.getInt16(offset, true)
				channelData[i] = sample / 32768.0 // Convert to float
			}
		}

		return audioBuffer
	}

	// Keep Alive Management
	startKeepAlive() {
		this.stopKeepAlive()
		this.keepAliveInterval = setInterval(() => {
			if (this.audioWs && this.audioWs.readyState === WebSocket.OPEN) {
				try {
					this.audioWs.send('ping')
					console.log('Sent audio keepalive ping')
				} catch (e) {
					console.log('Audio keepalive send error:', e)
				}
			}
		}, 20000)
	}

	stopKeepAlive() {
		if (this.keepAliveInterval) {
			clearInterval(this.keepAliveInterval)
			this.keepAliveInterval = null
		}
	}

	// Music Generation Control
	async startLiveProcessing() {
		this.liveBtn.disabled = true
		this.updateStatus(
			this.liveStatus,
			'Initializing AI music generation...',
			'processing'
		)

		try {
			// Connect video WebSocket first
			this.connectVideoWebSocket()
			const userSettings = {
				bpm: parseInt(this.bpmSlider.value),
				scale: this.scaleSelect.value,
				genre: this.genreInput.value.trim(),
			}

			// Start music processing on server
			const response = await fetch('/start-live', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify(userSettings),
			})

			if (!response.ok) {
				const errorText = await response.text()
				throw new Error(errorText || 'Failed to start music generation')
			}

			const result = await response.json()
			this.updateStatus(
				this.liveStatus,
				result.message || 'Live music generation active',
				'connected'
			)
			this.stopMusicBtn.disabled = false
		} catch (error) {
			console.error('Live processing error:', error)
			this.updateStatus(
				this.liveStatus,
				error.message || 'Failed to start music generation',
				'error'
			)
			this.liveBtn.disabled = false
		}
	}

	async stopMusicGeneration() {
		try {
			// Stop server-side processing first
			const response = await fetch('/stop-live', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
			})

			if (response.ok) {
				console.log('Server processing stopped')
			}
		} catch (error) {
			console.error('Error stopping server processing:', error)
		}

		// Close WebSocket connections
		if (this.videoWs) {
			this.videoWs.close()
			this.videoWs = null
		}

		if (this.frameInterval) {
			clearInterval(this.frameInterval)
			this.frameInterval = null
		}

		// Reset UI
		this.liveBtn.disabled = false
		this.stopMusicBtn.disabled = true

		this.updateStatus(
			this.liveStatus,
			'Music generation stopped',
			'processing'
		)
	}

	// Utility Methods
	updateStatus(element, message, type = '') {
		if (!element) return

		element.textContent = message

		// Update corresponding status dot with animation
		let statusDot = null
		if (element === this.connectionStatus) {
			statusDot = this.audioStatusDot
		} else if (element === this.cameraStatus) {
			statusDot = this.cameraStatusDot
		} else if (element === this.liveStatus) {
			statusDot = this.musicStatusDot
		}

		if (statusDot) {
			statusDot.className = `status-dot ${type}`
			// Add visual feedback animation
			statusDot.style.transform = 'scale(1.2)'
			setTimeout(() => {
				statusDot.style.transform = 'scale(1)'
			}, 200)
		}
	}

	cleanup() {
		this.stopKeepAlive()

		// Close WebSocket connections
		if (this.audioWs) {
			this.audioWs.close()
		}

		if (this.videoWs) {
			this.videoWs.close()
		}

		// Stop media stream
		if (this.mediaStream) {
			this.mediaStream.getTracks().forEach((track) => track.stop())
		}

		// Stop audio source
		if (this.audioSource) {
			this.audioSource.stop()
		}

		// Clear intervals
		if (this.visualizationInterval) {
			clearInterval(this.visualizationInterval)
		}

		if (this.frameInterval) {
			clearInterval(this.frameInterval)
		}

		console.log('Application cleanup completed')
	}
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
	console.log('Initializing Live Music Generator Pro...')
	window.liveMusicGenerator = new LiveMusicGenerator()
})
