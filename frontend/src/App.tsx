// src/App.tsx
import { useState, useEffect } from 'react';
import { healthCheck } from './lib/api';
import Chat from './components/Chat';
import './components/Chat.css';

interface HealthStatus {
  status: string;
  app: string;
  model: string;
}

function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showChat, setShowChat] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        setLoading(true);
        setError(null);
        const result = await healthCheck();
        setHealth(result);
        // Automatically show chat if backend is healthy
        setShowChat(true);
      } catch (err) {
        console.error('Health check failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to connect to API');
        setShowChat(false);
      } finally {
        setLoading(false);
      }
    };

    checkHealth();
  }, []);

  // Show chat interface if backend is connected
  if (showChat && health && !error) {
    return <Chat />;
  }

  // Show connection status screen
  return (
    <div className="connection-screen">
      <div className="connection-container">
        <h1>AI Medical Chatbot</h1>
        
        <div className="health-status">
          <h2>Backend Connection Status</h2>
          
          {loading && (
            <div className="status loading">
              <div className="status-icon">⏳</div>
              <p>Checking connection...</p>
            </div>
          )}
          
          {error && (
            <div className="status error">
              <div className="status-icon">❌</div>
              <p><strong>Connection Failed</strong></p>
              <p className="error-message">{error}</p>
              <button onClick={() => window.location.reload()} className="retry-button">
                Retry Connection
              </button>
            </div>
          )}
          
          {health && !loading && !error && (
            <div className="status success">
              <div className="status-icon">✅</div>
              <p><strong>Backend Connected</strong></p>
              <div className="health-details">
                <p><strong>Status:</strong> {health.status}</p>
                <p><strong>App:</strong> {health.app}</p>
                <p><strong>Model:</strong> {health.model}</p>
              </div>
              <button onClick={() => setShowChat(true)} className="start-chat-button">
                Start Chat
              </button>
            </div>
          )}
        </div>
        
        {health && (
          <div className="next-steps">
            <h3>Ready for Medical Assistance!</h3>
            <p>Your AI medical assistant is ready to help with health-related questions.</p>
            <p className="disclaimer-note">
              <strong>Important:</strong> This AI provides general health information only. 
              Always consult healthcare professionals for medical advice.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;