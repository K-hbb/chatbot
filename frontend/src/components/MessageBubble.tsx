// src/components/MessageBubble.tsx
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import SourceBadge from './SourceBadge.tsx';
import type { Source } from '../lib/api';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  emergency?: boolean;
  disclaimer?: string;
  timestamp: Date;
}

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble = ({ message }: MessageBubbleProps) => {
  const [showAllSources, setShowAllSources] = useState(false);
  
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const displaySources = message.sources || [];
  const visibleSources = showAllSources ? displaySources : displaySources.slice(0, 3);
  const hasMoreSources = displaySources.length > 3;

  return (
    <div className={`message-bubble ${message.type}-message`}>
      <div className="message-header">
        <span className="message-sender">
          {message.type === 'user' ? 'You' : 'AI Medical Assistant'}
        </span>
        <span className="message-time">
          {formatTime(message.timestamp)}
        </span>
      </div>

      <div className="message-content">
        {/* Emergency banner for assistant messages */}
        {message.type === 'assistant' && message.emergency && (
          <div className="emergency-banner">
            <span className="emergency-icon">🚨</span>
            <strong>Emergency Notice:</strong> If you are experiencing a medical emergency, 
            call your local emergency number immediately.
          </div>
        )}

        {/* Message text */}
        <div className="message-text">
          {message.type === 'user' ? (
            <p>{message.content}</p>
          ) : (
            <ReactMarkdown
              components={{
                // Customize how links render
                a: ({ href, children }) => (
                  <a href={href} target="_blank" rel="noopener noreferrer" className="external-link">
                    {children}
                  </a>
                ),
                // Customize code blocks
                code: ({ className, children }) => (
                  <code className={`inline-code ${className || ''}`}>
                    {children}
                  </code>
                ),
                pre: ({ children }) => (
                  <pre className="code-block">{children}</pre>
                ),
                // Customize lists
                ul: ({ children }) => (
                  <ul className="markdown-list">{children}</ul>
                ),
                ol: ({ children }) => (
                  <ol className="markdown-list">{children}</ol>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {/* Sources section for assistant messages */}
        {message.type === 'assistant' && displaySources.length > 0 && (
          <div className="sources-section">
            <h4 className="sources-title">Sources:</h4>
            <div className="sources-container">
              {visibleSources.map((source, index) => (
                <SourceBadge key={`${source.id}-${index}`} source={source} index={index + 1} />
              ))}
              
              {hasMoreSources && !showAllSources && (
                <button
                  onClick={() => setShowAllSources(true)}
                  className="show-more-sources"
                >
                  +{displaySources.length - 3} more sources
                </button>
              )}
              
              {hasMoreSources && showAllSources && (
                <button
                  onClick={() => setShowAllSources(false)}
                  className="show-fewer-sources"
                >
                  Show fewer
                </button>
              )}
            </div>
          </div>
        )}

        {/* Disclaimer for assistant messages */}
        {message.type === 'assistant' && message.disclaimer && (
          <div className="message-disclaimer">
            <small>{message.disclaimer}</small>
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;