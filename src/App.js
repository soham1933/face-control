import React from 'react';

function App() {
  return (
    <div className="App">
      <h1>Face Tracking Pointer</h1>
      <img
        src="http://127.0.0.1:5000/track"
        alt="Live Video Stream"
        style={{ width: '100%', height: 'auto' }}
      />
    </div>
  );
}

export default App;
