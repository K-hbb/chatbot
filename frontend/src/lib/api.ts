// src/lib/api.ts
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types matching your backend models
export interface ChatRequest {
  question: string;
  conversation_id?: string;
}

export interface Source {
  id: string;
  score: number;
  title?: string;
  metadata: Record<string, any>;
}

export interface ChatResponse {
  text: string;
  sources: Source[];
  emergency: boolean;
  disclaimer?: string;
}

export interface SearchRequest {
  query: string;
  k?: number;
}

export interface SearchHit {
  doc_id: string;
  title?: string;
  snippet: string;
  score: number;
  metadata: Record<string, any>;
}

export interface SearchResponse {
  hits: SearchHit[];
}

// Streaming message types
export interface StreamMessage {
  type: 'safety' | 'sources' | 'text' | 'done' | 'error';
  emergency?: boolean;
  disclaimer?: string;
  data?: any;
  message?: string;
}

// API functions
export const healthCheck = async (): Promise<{ status: string; app: string; model: string }> => {
  const response = await api.get('/health');
  return response.data;
};

export const search = async (req: SearchRequest): Promise<SearchResponse> => {
  const response = await api.post('/search', req);
  return response.data;
};

export const chatSync = async (req: ChatRequest): Promise<ChatResponse> => {
  const response = await api.post('/chat-sync', req);
  return response.data;
};

// Streaming chat function
export const chatStream = async (
  req: ChatRequest,
  onMessage: (message: StreamMessage) => void,
  onComplete: () => void,
  onError: (error: string) => void
): Promise<void> => {
  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body reader available');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      
      // Process complete SSE messages
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const jsonStr = line.slice(6); // Remove 'data: '
            if (jsonStr.trim()) {
              const message: StreamMessage = JSON.parse(jsonStr);
              onMessage(message);
              
              if (message.type === 'done') {
                onComplete();
                return;
              } else if (message.type === 'error') {
                onError(message.message || 'Unknown streaming error');
                return;
              }
            }
          } catch (e) {
            console.warn('Failed to parse SSE message:', line, e);
          }
        }
      }
    }
    
    onComplete();
  } catch (error) {
    console.error('Streaming error:', error);
    onError(error instanceof Error ? error.message : 'Unknown error');
  }
};