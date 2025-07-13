import React, { useState, useCallback } from 'react';
import SearchTreeViewer from './components/SearchTreeViewer';

const App: React.FC = () => {
  const [logText, setLogText] = useState<string>('');
  const [fileName, setFileName] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setFileName(file.name);
      setIsLoading(true);
      setError('');
      
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = e.target?.result as string;
          setLogText(content);
          setIsLoading(false);
        } catch (err) {
          setError('Failed to read file');
          setIsLoading(false);
        }
      };
      reader.onerror = () => {
        setError('Error reading file');
        setIsLoading(false);
      };
      reader.readAsText(file);
    }
  }, []);

  const handleTextareaChange = useCallback((event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setLogText(event.target.value);
    setFileName('');
    setError('');
  }, []);

  const handleClearLogs = useCallback(() => {
    setLogText('');
    setFileName('');
    setError('');
  }, []);

  const sampleLogText = `{"event": "node_expansion", "node_id": "node_1_rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-", "parent_id": null, "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1234, "U": 0.0567, "expval": 1.1234, "expoppval": 0.8766, "is_terminal": false, "potential_children": [{"move": "e2e4", "move_san": "e4", "probability": 0.3456, "U": 0.0123, "Q": 0.1234, "D": 0.0567, "expanded": true, "child_id": "node_2_rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR_b_KQkq_e3"}, {"move": "d2d4", "move_san": "d4", "probability": 0.2345, "U": 0.0234, "Q": 0.0987, "D": 0.0654, "expanded": false, "child_id": null}], "num_potential_children": 2, "timestamp": 1}
{"event": "node_expansion", "node_id": "node_2_rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR_b_KQkq_e3", "parent_id": "node_1_rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-", "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "value": -0.0987, "U": 0.0432, "expval": 0.9012, "expoppval": 1.0987, "is_terminal": false, "potential_children": [{"move": "e7e5", "move_san": "e5", "probability": 0.4567, "U": 0.0345, "Q": -0.0987, "D": 0.0432, "expanded": true, "child_id": "node_3_rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR_w_KQkq_e6"}, {"move": "d7d5", "move_san": "d5", "probability": 0.2987, "U": 0.0234, "Q": -0.0654, "D": 0.0321, "expanded": false, "child_id": null}], "num_potential_children": 2, "timestamp": 2}
{"event": "node_expansion", "node_id": "node_3_rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR_w_KQkq_e6", "parent_id": "node_2_rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR_b_KQkq_e3", "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "value": 0.0654, "U": 0.0543, "expval": 1.0654, "expoppval": 0.9346, "is_terminal": false, "potential_children": [{"move": "g1f3", "move_san": "Nf3", "probability": 0.3876, "U": 0.0123, "Q": 0.0654, "D": 0.0543, "expanded": false, "child_id": null}, {"move": "f1c4", "move_san": "Bc4", "probability": 0.2543, "U": 0.0234, "Q": 0.0321, "D": 0.0432, "expanded": false, "child_id": null}], "num_potential_children": 2, "timestamp": 3}`;

  const handleLoadSample = useCallback(() => {
    setLogText(sampleLogText);
    setFileName('sample_data.json');
    setError('');
  }, []);

  if (logText.trim()) {
    return <SearchTreeViewer logText={logText} />;
  }

  return (
    <div style={{ 
      padding: '40px', 
      maxWidth: '800px', 
      margin: '0 auto', 
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' 
    }}>
      <h1 style={{ 
        textAlign: 'center', 
        color: '#212529', 
        marginBottom: '40px' 
      }}>
        Chess Search Tree Viewer
      </h1>
      
      <div style={{ 
        backgroundColor: '#ffffff', 
        borderRadius: '8px', 
        padding: '30px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        border: '1px solid #dee2e6'
      }}>
        <h2 style={{ marginTop: '0', color: '#495057' }}>Load Search Logs</h2>
        
        <div style={{ marginBottom: '20px' }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '10px', 
            fontWeight: 'bold', 
            color: '#495057' 
          }}>
            Upload Log File:
          </label>
          <input
            type="file"
            accept=".txt,.log,.json"
            onChange={handleFileChange}
            disabled={isLoading}
            style={{
              padding: '10px',
              border: '1px solid #ced4da',
              borderRadius: '4px',
              backgroundColor: isLoading ? '#f8f9fa' : '#ffffff',
              width: '100%',
              cursor: isLoading ? 'not-allowed' : 'pointer'
            }}
          />
          {fileName && (
            <p style={{ 
              margin: '10px 0 0 0', 
              color: '#28a745', 
              fontSize: '14px' 
            }}>
              Loaded: {fileName}
            </p>
          )}
        </div>

        <div style={{ 
          textAlign: 'center', 
          margin: '20px 0',
          color: '#6c757d'
        }}>
          — or —
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '10px', 
            fontWeight: 'bold', 
            color: '#495057' 
          }}>
            Paste Log Text:
          </label>
          <textarea
            value={logText}
            onChange={handleTextareaChange}
            placeholder="Paste your search log JSON lines here..."
            rows={10}
            style={{
              width: '100%',
              padding: '10px',
              border: '1px solid #ced4da',
              borderRadius: '4px',
              fontFamily: 'monospace',
              fontSize: '14px',
              resize: 'vertical',
              backgroundColor: '#ffffff'
            }}
          />
        </div>

        <div style={{ 
          display: 'flex', 
          gap: '10px', 
          justifyContent: 'center',
          marginBottom: '20px'
        }}>
          <button
            onClick={handleLoadSample}
            style={{
              padding: '10px 20px',
              backgroundColor: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            Load Sample Data
          </button>
          {logText && (
            <button
              onClick={handleClearLogs}
              style={{
                padding: '10px 20px',
                backgroundColor: '#dc3545',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              Clear Logs
            </button>
          )}
        </div>

        {isLoading && (
          <div style={{ 
            textAlign: 'center', 
            color: '#007bff',
            fontStyle: 'italic'
          }}>
            Loading file...
          </div>
        )}

        {error && (
          <div style={{ 
            textAlign: 'center', 
            color: '#dc3545',
            backgroundColor: '#f8d7da',
            padding: '10px',
            borderRadius: '4px',
            border: '1px solid #f5c6cb'
          }}>
            {error}
          </div>
        )}

        <div style={{ 
          marginTop: '30px', 
          padding: '20px',
          backgroundColor: '#f8f9fa',
          borderRadius: '4px',
          border: '1px solid #e9ecef'
        }}>
          <h3 style={{ marginTop: '0', color: '#495057' }}>How to Use</h3>
          <ol style={{ color: '#6c757d', lineHeight: '1.6' }}>
            <li>Run your chess engine with verbose logging enabled</li>
            <li>Capture the JSON output from the search tree logging</li>
            <li>Upload the log file or paste the JSON lines above</li>
            <li>Click on nodes in the tree to view detailed information</li>
            <li>Use zoom and pan to navigate large trees</li>
          </ol>
          <p style={{ color: '#6c757d', fontSize: '14px', marginBottom: '0' }}>
            Expected format: One JSON object per line with node expansion events.
          </p>
        </div>
      </div>
    </div>
  );
};

export default App; 