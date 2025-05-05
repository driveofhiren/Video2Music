import React, { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
	const [analysis, setAnalysis] = useState(null)
	const [music, setMusic] = useState(null)
	const videoRef = useRef(null)
	const ws = useRef(null)

	useEffect(() => {
		// Connect to WebSocket
		ws.current = new WebSocket('ws://localhost:8000/ws')

		ws.current.onmessage = (event) => {
			const data = JSON.parse(event.data)
			setAnalysis(data.analysis)
			setMusic(data.music)

			// Display video frame
			if (videoRef.current) {
				const imgData = new Uint8Array(Buffer.from(data.frame, 'hex'))
				const blob = new Blob([imgData], { type: 'image/jpeg' })
				videoRef.current.src = URL.createObjectURL(blob)
			}
		}

		return () => ws.current.close()
	}, [])

	return (
		<div className="container">
			<div className="video-container">
				<img ref={videoRef} alt="Live camera feed" />
				{analysis && (
					<div className="overlay">
						<p>Mood: {analysis.mood}</p>
						<p>BPM: {music?.bpm}</p>
						<p>Scale: {analysis.scale}</p>
					</div>
				)}
			</div>

			{music && (
				<div className="music-display">
					<h3>Generated Melody</h3>
					<div className="notes">
						{music.melody.map((note, i) => (
							<div
								key={i}
								className="note"
								style={{
									height: `${note.note - 50}px`,
									width: `${note.duration * 50}px`,
								}}
							></div>
						))}
					</div>
				</div>
			)}
		</div>
	)
}

export default App
