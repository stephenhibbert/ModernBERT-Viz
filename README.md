# 🧠 ModernBERT Token Analysis

An interactive web application for visualizing how token embeddings evolve through the layers of ModernBERT and exploring attention patterns in real-time.

![ModernBERT Analysis Interface](screenshot.png)

## 🎯 What is this?

This project provides an intuitive way to understand how transformer models process language by visualizing:

- **Token Trajectories**: How individual tokens move through high-dimensional embedding space across layers (projected to 2D using PCA)
- **Attention Patterns**: Interactive heatmaps showing which tokens attend to which others at different layers
- **Model Architecture**: Live view of ModernBERT's structure with 149M parameters broken down by component

## ✨ Key Features

### 🔍 **Token Trajectory Analysis**
- Select up to 6 tokens from any sentence
- Watch how embeddings evolve through all 22 layers
- Compare different token types (content words, function words, punctuation)
- PCA projection shows 2D representation of high-dimensional transformations

### 🔥 **Attention Visualization**
- Interactive heatmaps for attention patterns
- Layer-by-layer exploration (every 3rd layer sampled for performance)
- See which tokens the model "pays attention to"
- Synchronized with architecture panel highlighting

### 🏗️ **Live Architecture View**
- Real-time model structure inspection
- 149M parameter breakdown by component
- Highlights current layer being analyzed
- Detailed view of attention and MLP components

### 📱 **Modern Interface**
- Split-screen design for simultaneous analysis
- Responsive layout (desktop/tablet/mobile)
- Interactive token selection with visual feedback
- Real-time processing with loading indicators

## 🚀 Quick Start

### Prerequisites
```bash
pip install fastapi uvicorn transformers torch scikit-learn numpy
```

### Run the Application
```bash
python app.py
```

Then open your browser to `http://localhost:8000`

### Basic Usage
1. **Enter a sentence** (or use the default: "The cat sat on the mat.")
2. **Click "Tokenize Sentence"** to see available tokens
3. **Select 1-6 tokens** by clicking on them (they'll show selection order)
4. **Click "Analyze Tokens"** to generate visualizations
5. **Explore layers** using the attention heatmap controls

## 🔬 What You Can Discover

### Token Behavior Patterns
- **Content words** (nouns, verbs) often cluster together in embedding space
- **Function words** (articles, prepositions) may follow different trajectories
- **Punctuation** tokens often have unique embedding patterns
- **Layer progression** shows how representations become more refined

### Attention Insights
- **Early layers** often show more distributed attention
- **Later layers** may focus on specific syntactic/semantic relationships
- **Different heads** in the same layer can specialize in different patterns
- **Self-attention** vs. **cross-attention** patterns

### Architecture Understanding
- **Parameter distribution**: Where ModernBERT's 149M parameters are allocated
- **Component sizing**: Attention vs. MLP parameter ratios
- **Layer structure**: How the 22 identical encoder layers are organized

## 🛠️ Technical Details

### Model
- **ModernBERT-base**: 22 layers, 12 attention heads, 768 hidden dimensions
- **Vocabulary**: 50,368 tokens
- **Architecture**: Encoder-only transformer with rotary position embeddings
- **Attention**: Eager implementation for attention weight extraction

### Visualization
- **PCA**: Dimensionality reduction from 768D to 2D for trajectory plots
- **Plotly.js**: Interactive plotting with hover details and zoom
- **FastAPI**: High-performance backend with automatic API documentation
- **Responsive Design**: CSS Grid/Flexbox for adaptive layouts

### Performance Optimizations
- **Layer sampling**: Shows every 3rd layer for attention (0, 3, 6, 9, ...)
- **Efficient parameter counting**: Avoids double-counting in architecture analysis
- **Client-side caching**: Tokenization results cached for multiple analyses
- **Streaming responses**: Real-time feedback during processing

## 📚 Educational Use Cases

### For Students
- **Understand transformers**: See how attention and embeddings actually work
- **Compare token types**: Observe different linguistic patterns
- **Layer analysis**: Watch representations evolve through the network

### For Researchers
- **Model interpretability**: Gain insights into ModernBERT's internal representations
- **Attention analysis**: Study attention patterns across layers and heads
- **Embedding dynamics**: Analyze how different tokens behave in embedding space

### For Developers
- **Model debugging**: Understand why certain predictions are made
- **Architecture exploration**: See how parameters are distributed
- **Visualization techniques**: Learn interactive ML visualization approaches

## 🎮 Example Experiments

### Compare Word Types
```
Sentence: "The quick brown fox jumps over the lazy dog"
Select: "The" (article), "quick" (adjective), "fox" (noun), "jumps" (verb)
Observe: How different parts of speech move through embedding space
```

### Analyze Attention Patterns
```
Sentence: "The cat that I saw yesterday was sleeping"
Focus: Layer 12 attention patterns
Look for: Subject-verb relationships, relative clause attention
```

### Study Punctuation
```
Sentence: "Hello, world! How are you today?"
Select: "Hello", ",", "!", "?"
Compare: Content vs. punctuation token trajectories
```

## 🤝 Contributing

This project demonstrates transformer interpretability techniques. Contributions welcome for:
- Additional visualization types
- Performance optimizations  
- New analysis features
- Educational content

## 📄 License

MIT License - Feel free to use for educational and research purposes.

## 🙏 Acknowledgments

- **ModernBERT**: Answer.AI and LightOn for the model
- **Transformers**: Hugging Face for the excellent library
- **Visualization**: Plotly.js for interactive graphics
- **Community**: ML interpretability research community

---

*Explore the inner workings of transformer models through interactive visualization!* 🚀