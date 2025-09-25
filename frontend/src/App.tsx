import { useState, useEffect } from 'react';
import { healthCheck } from './lib/api';
import './App.css';

interface HealthStatus {
  status: string;
  app: string;
  model: string;
}

function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        setLoading(true);
        setError(null);
        const result = await healthCheck();
        setHealth(result);
      } catch (err) {
        console.error('Health check failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to connect to API');
      } finally {
        setLoading(false);
      }
    };

    checkHealth();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Medical Chatbot</h1>
        
        <div className="health-status">
          <h2>Backend Connection Status</h2>
          
          {loading && (
            <div className="status loading">
              <p>Checking connection...</p>
            </div>
          )}
          
          {error && (
            <div className="status error">
              <p>❌ Connection Failed</p>
              <p className="error-message">{error}</p>
              <button onClick={() => window.location.reload()}>
                Retry Connection
              </button>
            </div>
          )}
          
          {health && !loading && !error && (
            <div className="status success">
              <p>✅ Backend Connected</p>
              <div className="health-details">
                <p><strong>Status:</strong> {health.status}</p>
                <p><strong>App:</strong> {health.app}</p>
                <p><strong>Model:</strong> {health.model}</p>
              </div>
            </div>
          )}
        </div>
        
        {health && (
          <div className="next-steps">
            <h3>Ready for Chat Interface!</h3>
            <p>Backend is running and ready to receive chat requests.</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;