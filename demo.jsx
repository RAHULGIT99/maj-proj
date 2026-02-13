// components/ChatInterface.jsx
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Download } from 'lucide-react';
import './ChatInterface.css';

const ChatInterface = ({ indexName, summary, messages, onMessagesChange, activeTab }) => {
  const [question, setQuestion] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  
  const bottomRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    // 1. Add User Message
    const userMsg = { id: Date.now(), role: 'user', text: question };
    const updatedMessages = [...messages, userMsg];
    onMessagesChange(updatedMessages);

    setQuestion("");
    setIsAsking(true);

    try {
      const token = localStorage.getItem('token');
      const config = { headers: {} };
      if (token) config.headers['Authorization'] = `Bearer ${token}`;

      // Build conversation history from existing messages (last 10, text only)
      const history = messages.slice(-10).map(m => ({
        role: m.role,
        text: m.text
      }));

      let endpoint = 'http://localhost:8000/ask';
      let payload = {};
      let aiMsg = { id: Date.now() + 1, role: 'ai', text: "" };

      // --- LOGIC PER TAB ---
      if (activeTab === 'viz') {
          endpoint = 'http://localhost:8000/visuals';
          payload = {
              query: userMsg.text,
              index: indexName,
              history: history
          };

          const response = await axios.post(endpoint, payload, config);

          if (response.data.response_type === 'viz') {
            // Visualization was generated
            aiMsg.text = response.data.message || `Generated ${response.data.visualization_type}: ${response.data.task}`;
            aiMsg.images = response.data.images;
            aiMsg.isVisual = true;
          } else {
            // Chat response (no chart)
            aiMsg.text = response.data.message;
          }

      } else if (activeTab === 'excel') {
          endpoint = 'http://localhost:8000/excel';
          payload = { query: userMsg.text, index: indexName, history: history };
          const response = await axios.post(endpoint, payload, config);

          if (response.data.response_type === 'excel') {
            // Excel file was generated
            aiMsg.text = response.data.message || 'Excel file generated.';
            aiMsg.isExcel = true;
            aiMsg.excelBase64 = response.data.file_base64;
            aiMsg.excelFilename = response.data.filename || 'export.xlsx';
          } else {
            // Chat response (no file)
            aiMsg.text = response.data.message;
          }

      } else {
          // Default: Chat / Summary
          endpoint = 'http://localhost:8000/ask';
          payload = { index_name: indexName, question: userMsg.text, history: history };
          const response = await axios.post(endpoint, payload, config);
          aiMsg.text = response.data.answer;
      }

      // Update state with the new AI message
      onMessagesChange([...updatedMessages, aiMsg]);

    } catch (error) {
      console.error("Chat Error", error);
      onMessagesChange([...updatedMessages, { 
        id: Date.now() + 1, 
        role: 'ai', 
        text: "Error: Could not fetch result. Please try again." 
      }]);
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="chat-container">
      {/* Messages Area */}
      <div className="messages-area">
        
        {/* Render Summary ONLY for Chat Tab */}
        {activeTab === 'chat' && summary && (
          <div className="message-wrapper ai">
            <div className="message-bubble ai">
              <div style={{whiteSpace: 'pre-wrap'}}>
                <strong>Analysis Complete.</strong>
                <br /><br />
                Here is the summary:
                <br />
                {summary}
              </div>
            </div>
          </div>
        )}

        {/* Message Loop */}
        {messages.map((msg) => (
          <div 
            key={msg.id} 
            className={`message-wrapper ${msg.role === 'user' ? 'user' : 'ai'}`}
          >
            <div className={`message-bubble ${msg.role === 'user' ? 'user' : 'ai'}`}>
              
              {/* Text Content */}
              <div style={{whiteSpace: 'pre-wrap'}}>{msg.text}</div>

              {/* Visualization Images (Only if present) */}
              {msg.isVisual && msg.images && (
                <div className="viz-images-container" style={{ marginTop: '10px' }}>
                  {msg.images.map((imgStr, idx) => (
                    <img 
                      key={idx}
                      src={`data:image/png;base64,${imgStr}`} 
                      alt={`Visualization ${idx}`}
                      style={{ maxWidth: '100%', borderRadius: '8px', border: '1px solid #ddd' }}
                    />
                  ))}
                </div>
              )}

              {/* Excel Download Button */}
              {msg.isExcel && msg.excelBase64 && (
                <button
                  onClick={() => {
                    const byteChars = atob(msg.excelBase64);
                    const byteNums = new Array(byteChars.length);
                    for (let i = 0; i < byteChars.length; i++) {
                      byteNums[i] = byteChars.charCodeAt(i);
                    }
                    const byteArray = new Uint8Array(byteNums);
                    const blob = new Blob([byteArray], {
                      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = msg.excelFilename || 'export.xlsx';
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  style={{
                    marginTop: '10px',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '8px 16px',
                    background: '#10b981',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontSize: '14px',
                  }}
                >
                  <Download size={16} /> Download Excel
                </button>
              )}

            </div>
          </div>
        ))}

        {isAsking && (
           <div className="thinking-indicator-wrapper">
             <div className="thinking-indicator">
               Processing...
             </div>
           </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input Area */}
      <form onSubmit={handleAsk} className="input-area">
        <input 
          type="text" 
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={activeTab === 'viz' ? "Describe the chart you want..." : activeTab === 'excel' ? "Describe the data you want in Excel..." : "Ask a question..."}
          disabled={isAsking}
          className="chat-input"
        />
        <button 
          type="submit" 
          disabled={isAsking || !question.trim()}
          className="send-btn"
        >
          <Send size={20} />
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;