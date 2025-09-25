// src/components/SourceBadge.tsx
import { useState } from 'react';
import type { Source } from '../lib/api';

interface SourceBadgeProps {
  source: Source;
  index: number;
}

const SourceBadge = ({ source, index }: SourceBadgeProps) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Extract useful information from metadata
  const getDisplayTitle = () => {
    if (source.title) return source.title;
    if (source.metadata?.title) return source.metadata.title;
    if (source.metadata?.source_path) {
      const filename = source.metadata.source_path.split('/').pop() || source.metadata.source_path;
      return filename.replace(/\.[^/.]+$/, ""); // Remove extension
    }
    return `Source ${index}`;
  };

  const getContentType = () => {
    return source.metadata?.content_type || 'document';
  };

  const getSnippet = () => {
    // Try to get a snippet from various possible fields
    if (source.metadata?.snippet) return source.metadata.snippet;
    if (source.metadata?.question) {
      // For Q&A pairs, show the question
      return `Q: ${source.metadata.question}`;
    }
    return null;
  };

  const formatScore = (score: number) => {
    return Math.round(score * 100);
  };

  const getScoreColor = (score: number) => {
    if (score > 0.8) return 'high-relevance';
    if (score > 0.6) return 'medium-relevance';
    return 'low-relevance';
  };

  const snippet = getSnippet();
  const displayTitle = getDisplayTitle();
  const contentType = getContentType();
  const scoreColor = getScoreColor(source.score);

  return (
    <div className={`source-badge ${scoreColor}`}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="source-badge-button"
        aria-expanded={isExpanded}
      >
        <div className="source-badge-header">
          <span className="source-index">[{index}]</span>
          <span className="source-title">{displayTitle}</span>
          <span className="source-score">{formatScore(source.score)}%</span>
          <span className={`expand-icon ${isExpanded ? 'expanded' : ''}`}>
            ▼
          </span>
        </div>
        
        <div className="source-meta">
          <span className="content-type">{contentType}</span>
          {source.metadata?.source_path && (
            <span className="source-file">
              📄 {source.metadata.source_path}
            </span>
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="source-details">
          {snippet && (
            <div className="source-snippet">
              <h5>Preview:</h5>
              <p>{snippet}</p>
            </div>
          )}
          
          {/* Show additional metadata for Q&A pairs */}
          {contentType === 'medical_qa' && source.metadata?.answer && (
            <div className="source-snippet">
              <h5>Answer Preview:</h5>
              <p>{source.metadata.answer.substring(0, 200)}...</p>
            </div>
          )}
          
          <div className="source-metadata">
            <h5>Details:</h5>
            <div className="metadata-grid">
              <div className="metadata-item">
                <strong>Relevance:</strong> {formatScore(source.score)}%
              </div>
              {source.metadata?.qa_pair_index !== undefined && (
                <div className="metadata-item">
                  <strong>Q&A Pair:</strong> #{source.metadata.qa_pair_index + 1}
                </div>
              )}
              {source.metadata?.chunk_index !== undefined && (
                <div className="metadata-item">
                  <strong>Section:</strong> {source.metadata.chunk_index + 1}
                  {source.metadata?.total_chunks && ` of ${source.metadata.total_chunks}`}
                </div>
              )}
              <div className="metadata-item">
                <strong>Source ID:</strong> <code>{source.id}</code>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SourceBadge;