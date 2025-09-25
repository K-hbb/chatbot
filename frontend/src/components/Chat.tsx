// src/components/Chat.tsx
import { useState, useRef, useEffect } from 'react';
import  { chatStream } from '../lib/api';
import type { Source } from '../lib/api';
import type { StreamMessage } from '../lib/api';
import MessageBubble from './MessageBubble';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  emergency?: boolean;
  disclaimer?: string;
  timestamp: Date;
}

const Chat = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState<string>('');
  const [currentSources, setCurrentSources] = useState<Source[]>([]);
  const [currentEmergency, setCurrentEmergency] = useState(false);
  const [currentDisclaimer, setCurrentDisclaimer] = useState<string>('');
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentStreamingMessage]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const question = inputValue.trim();
    if (!question || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: question,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    
    // Reset streaming state
    setCurrentStreamingMessage('');
    setCurrentSources([]);
    setCurrentEmergency(false);
    setCurrentDisclaimer('');

    // Use refs to capture the final state values
    let finalMessage = '';
    let finalSources: Source[] = [];
    let finalEmergency = false;
    let finalDisclaimer = '';

    // Start streaming chat
    await chatStream(
      { question },
      (message: StreamMessage) => {
        switch (message.type) {
          case 'safety':
            finalEmergency = message.emergency || false;
            finalDisclaimer = message.disclaimer || '';
            setCurrentEmergency(finalEmergency);
            setCurrentDisclaimer(finalDisclaimer);
            break;
          
          case 'sources':
            if (message.data) {
              finalSources = message.data as Source[];
              setCurrentSources(finalSources);
            }
            break;
          
          case 'text':
            if (message.data) {
              finalMessage += message.data;
              setCurrentStreamingMessage(finalMessage);
            }
            break;
          
          case 'error':
            console.error('Streaming error:', message.message);
            finalMessage = finalMessage || `Error: ${message.message || 'Something went wrong'}`;
            setCurrentStreamingMessage(finalMessage);
            break;
        }
      },
      () => {
        // On complete - use the captured final values
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: finalMessage || 'I apologize, but I was unable to generate a response.',
          sources: finalSources,
          emergency: finalEmergency,
          disclaimer: finalDisclaimer,
          timestamp: new Date(),
        };
        
        setMessages(prev => [...prev, assistantMessage]);
        setIsLoading(false);
        setCurrentStreamingMessage('');
        setCurrentSources([]);
        setCurrentEmergency(false);
        setCurrentDisclaimer('');
      },
      (error: string) => {
        // On error
        console.error('Chat stream error:', error);
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: `I'm sorry, there was an error processing your request: ${error}`,
          timestamp: new Date(),
        };
        
        setMessages(prev => [...prev, errorMessage]);
        setIsLoading(false);
        setCurrentStreamingMessage('');
        setCurrentSources([]);
        setCurrentEmergency(false);
        setCurrentDisclaimer('');
      }
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>AI Medical Assistant</h2>
        <p className="chat-subtitle">
          Ask questions about health topics. This AI provides general information only.
        </p>
      </div>

      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-message">
            <div className="welcome-content">
              <h3>Welcome to your AI Medical Assistant</h3>
              <p>You can ask questions like:</p>
              <ul>
                <li>"What are the symptoms of the flu?"</li>
                <li>"How can I manage high blood pressure?"</li>
                <li>"When should I see a doctor for chest pain?"</li>
              </ul>
              <p className="disclaimer-text">
                Remember: This AI provides general health information only and cannot replace professional medical advice.
              </p>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {/* Show streaming message while assistant is typing */}
        {isLoading && (
          <div className="message-bubble assistant-message streaming">
            <div className="message-content">
              {currentEmergency && (
                <div className="emergency-banner">
                  <span className="emergency-icon">🚨</span>
                  <strong>Emergency Notice:</strong> If this is a medical emergency, call emergency services immediately.
                </div>
              )}
              
              {currentStreamingMessage ? (
                <div className="streaming-text">
                  {currentStreamingMessage}
                  <span className="typing-cursor">|</span>
                </div>
              ) : (
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                  AI is thinking...
                </div>
              )}
              
              {currentSources.length > 0 && (
                <div className="sources-preview">
                  <span className="sources-count">
                    Found {currentSources.length} relevant sources
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="chat-input-form">
        <div className="input-container">
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a medical question..."
            className="chat-input"
            rows={1}
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isLoading}
            className="send-button"
          >
            {isLoading ? (
              <span className="loading-spinner">⏳</span>
            ) : (
              <span className="send-icon">➤</span>
            )}
          </button>
        </div>
      </form>

      {/* Global disclaimer */}
      <div className="global-disclaimer">
        This AI provides general health information only and is not a substitute for professional medical advice. 
        Always consult a qualified clinician for diagnosis or treatment.
      </div>
    </div>
  );
};

export default Chat;s