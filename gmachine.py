import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import requests
import json
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import quote
import warnings
import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import threading
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GMachineAI:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
        })
        self.conversation_history = []
        self.search_cache = {}
        self.user_sessions = {}
        
    def intelligent_search(self, query):
        try:
            cache_key = query.lower().strip()
            if cache_key in self.search_cache:
                return self.search_cache[cache_key]
                
            encoded_query = quote(query)
            url = f"https://www.google.com/search?q={encoded_query}&num=5"
            
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for g in soup.find_all('div', class_='g'):
                try:
                    link = g.find('a')['href']
                    if link.startswith('/url?q='):
                        link = link[7:].split('&')[0]
                    
                    title_elem = g.find('h3')
                    snippet_elem = g.find('div', class_='VwiC3b')
                    
                    if title_elem and link and 'http' in link:
                        results.append({
                            'title': title_elem.get_text(),
                            'link': link,
                            'snippet': snippet_elem.get_text() if snippet_elem else "",
                            'source': 'Google'
                        })
                except Exception as e:
                    continue
            
            self.search_cache[cache_key] = results[:3]
            return results[:3]
            
        except Exception as e:
            logger.error(f"Arama hatası: {str(e)}")
            return []

    def get_web_content(self, url):
        try:
            response = self.session.get(url, timeout=8)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
                
            text_content = []
            for tag in ['p', 'h1', 'h2', 'h3', 'article']:
                elements = soup.find_all(tag)
                for elem in elements:
                    text = elem.get_text().strip()
                    if len(text) > 30:
                        text_content.append(text)
            
            content = ' '.join(text_content[:1000])
            return content if content else "İçerik alınamadı"
            
        except Exception as e:
            logger.error(f"İçerik alma hatası: {str(e)}")
            return "İçerik alınamadı"

    def generate_comprehensive_response(self, query, research_data):
        if not research_data:
            return "🔍 **Bu konuda yeterli bilgi bulunamadı.**\n\nLütfen farklı bir soru sorun veya sorunuzu daha net ifade edin.\n\n*Kaynak: G-Machine AI*"

        synthesized_response = "🚀 **G-Machine AI Yanıtı:**\n\n"
        
        main_points = []
        sources_info = []
        
        for i, result in enumerate(research_data[:3], 1):
            detailed_content = self.get_web_content(result['link'])
            
            summary = result['snippet'][:200] + "..." if len(result['snippet']) > 200 else result['snippet']
            main_points.append(f"• **{result['title']}**: {summary}")
            sources_info.append(f"{i}. {result['source']} - {result['link']}")

        synthesized_response += "\n".join(main_points)
        
        if sources_info:
            synthesized_response += f"\n\n📚 **Kaynaklar:**\n" + "\n".join(sources_info)
        
        synthesized_response += f"\n\n⚠️ **Önemli Not:** Yukarıdaki bilgiler çeşitli arama motorlarından derlenmiştir. Doğruluğunu teyit ediniz."

        query_lower = query.lower()
        if any(word in query_lower for word in ['python', 'kod', 'programlama', 'yazılım']):
            code_example = self.generate_python_example(query)
            synthesized_response += f"\n\n🐍 **Python Kod Örneği:**\n```python\n{code_example}\n```"
        
        elif any(word in query_lower for word in ['html', 'web', 'tasarım', 'css']):
            html_example = self.generate_html_example(query)
            synthesized_response += f"\n\n🌐 **HTML Kod Örneği:**\n```html\n{html_example}\n```"

        elif any(word in query_lower for word in ['veri', 'analiz', 'data', 'istatistik']):
            data_example = self.generate_data_analysis_example()
            synthesized_response += f"\n\n📊 **Veri Analiz Örneği:**\n```python\n{data_example}\n```"

        return synthesized_response

    def generate_python_example(self, query):
        examples = {
            'yapay zeka': '''
import torch
import torch.nn as nn

class GMachineAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = GMachineAI(784, 128, 10)
print("🤖 G-Machine AI Modeli hazır!")
''',
            'web': '''
import requests
from bs4 import BeautifulSoup
import json

class GMachineScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers = {
            'User-Agent': 'G-Machine-AI/1.0'
        }
    
    def search(self, query):
        try:
            url = f"https://www.google.com/search?q={query}"
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for item in soup.find_all('div', class_='g'):
                title = item.find('h3')
                if title:
                    results.append(title.get_text())
            
            return results[:5]
        except Exception as e:
            return f"Hata: {e}"

scraper = GMachineScraper()
print(scraper.search("G-Machine AI"))
'''
        }
        
        for key, code in examples.items():
            if key in query.lower():
                return code
        
        return '''
def g_machine_example():
    print("🚀 G-Machine AI Python Örneği")
    
    numbers = [1, 2, 3, 4, 5]
    squared = [x**2 for x in numbers]
    
    data = {
        'orjinal': numbers,
        'kareler': squared,
        'toplam': sum(squared)
    }
    
    print("Sonuç:", data)
    return data

result = g_machine_example()
'''

    def generate_html_example(self, query):
        return '''
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>G-Machine AI Sohbet</title>
    <style>
        :root {
            --primary: #8B5FBF;
            --secondary: #6D28D9;
            --dark: #1F2937;
            --darker: #111827;
            --light: #F3F4F6;
            --accent: #A78BFA;
        }
        
        body {
            font-family: 'Segoe UI', system-ui;
            background: linear-gradient(135deg, var(--darker), var(--dark));
            margin: 0;
            padding: 20px;
            min-height: 100vh;
            color: var(--light);
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>🚀 G-Machine AI</h1>
        <p>Gelişmiş Yapay Zeka Asistanı</p>
    </div>
</body>
</html>
'''

    def generate_data_analysis_example(self):
        return '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GMachineAnalyzer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
    
    def analyze(self):
        print("📊 G-Machine Veri Analizi")
        print("=" * 50)
        
        print("\\n🔍 İlk 5 satır:")
        print(self.data.head())
        
        print("\\n📈 Temel İstatistikler:")
        print(self.data.describe())
        
        print("\\n❌ Eksik Veriler:")
        print(self.data.isnull().sum())
        
        plt.figure(figsize=(12, 6))
        self.data.hist()
        plt.title("G-Machine Veri Dağılımı")
        plt.tight_layout()
        plt.show()

analyzer = GMachineAnalyzer('veriler.csv')
analyzer.analyze()
'''

    def process_query(self, query, user_id="default"):
        start_time = time.time()
        logger.info(f"Kullanıcı sorusu: {query}")
        
        search_results = self.intelligent_search(query)
        response = self.generate_comprehensive_response(query, search_results)
        
        processing_time = time.time() - start_time
        
        self.conversation_history.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'query': query,
            'response': response[:500] + "..." if len(response) > 500 else response,
            'processing_time': f"{processing_time:.1f}s"
        })
        
        logger.info(f"Yanıt hazır: {processing_time:.1f}s")
        return response

class AdvancedViT(nn.Module):
    def __init__(self, emb_size=192, depth=6, num_classes=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, emb_size, kernel_size=4, stride=4)
        num_patches = (28 // 4) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
        
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_size,
                nhead=6,
                dim_feedforward=emb_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.head(x[:, 0])

app = Flask(__name__)
gmachine_ai = GMachineAI()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="tr" class="h-full">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>G-Machine AI - Yapay Zeka Sohbet</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        wildberry: '#8B5A96',
                        'wildberry-dark': '#6B4576',
                        'wildberry-light': '#A66FB5',
                        'chat-bg': '#0f0f0f',
                        'sidebar-bg': '#1a1a1a',
                        'message-bg': '#262626',
                        'card-bg': '#1f1f1f'
                    },
                    animation: {
                        'fade-in': 'fadeIn 0.6s ease-out',
                        'slide-up': 'slideUp 0.5s ease-out',
                        'glow': 'glow 2s ease-in-out infinite alternate'
                    }
                }
            }
        }
    </script>
    <style>
        body {
            box-sizing: border-box;
        }
        
        .gradient-bg {
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #262626 100%);
        }
        
        .glass-effect {
            background: rgba(26, 26, 26, 0.8);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(139, 90, 150, 0.2);
        }
        
        .message-animation {
            animation: slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes glow {
            from { box-shadow: 0 0 20px rgba(139, 90, 150, 0.3); }
            to { box-shadow: 0 0 30px rgba(139, 90, 150, 0.6); }
        }
        
        .typing-dots {
            animation: typing 1.4s infinite;
        }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
            30% { transform: translateY(-8px); opacity: 1; }
        }
        
        .hover-lift {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .hover-lift:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(139, 90, 150, 0.3);
        }
        
        .code-block {
            background: linear-gradient(135deg, #1a1a1a, #0f0f0f);
            border: 1px solid #333;
            border-radius: 12px;
            padding: 16px;
            margin: 12px 0;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            overflow-x: auto;
            position: relative;
        }
        
        .code-block::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, #8B5A96, transparent);
        }
        
        .sidebar-item {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 8px;
        }
        
        .sidebar-item:hover {
            background: rgba(139, 90, 150, 0.15);
            transform: translateX(4px);
            border-left: 3px solid #8B5A96;
        }
        
        .chat-container {
            height: calc(100vh - 140px);
        }
        
        .message-user {
            background: linear-gradient(135deg, #8B5A96, #A66FB5);
            box-shadow: 0 4px 15px rgba(139, 90, 150, 0.3);
        }
        
        .message-ai {
            background: linear-gradient(135deg, #1f1f1f, #262626);
            border: 1px solid rgba(139, 90, 150, 0.2);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        .warning-box {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 193, 7, 0.05));
            border: 1px solid rgba(255, 193, 7, 0.3);
            border-radius: 8px;
            padding: 8px 12px;
            margin-top: 12px;
        }
        
        .input-glow:focus {
            box-shadow: 0 0 0 3px rgba(139, 90, 150, 0.3);
        }
        
        .pulse-glow {
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        .history-hidden {
            display: none;
        }
        
        .history-visible {
            display: block;
            animation: fadeIn 0.4s ease-out;
        }
    </style>
</head>
<body class="h-full gradient-bg text-white font-sans">
    <div class="flex h-full">
        <aside class="w-72 glass-effect flex flex-col">
            <div class="p-6 border-b border-gray-700/50">
                <div class="flex items-center space-x-4">
                    <div class="w-12 h-12 bg-gradient-to-br from-wildberry to-wildberry-light rounded-xl flex items-center justify-center pulse-glow">
                        <span class="text-2xl font-bold">۝</span>
                    </div>
                    <div>
                        <h1 class="text-xl font-bold text-white">G-Machine</h1>
                        <p class="text-sm text-gray-400">Powered by Gexnys</p>
                    </div>
                </div>
            </div>
            
            <div class="p-4">
                <button id="newChatBtn" class="w-full bg-gradient-to-r from-wildberry to-wildberry-light hover:from-wildberry-dark hover:to-wildberry text-white py-3 px-4 rounded-xl font-medium hover-lift flex items-center justify-center space-x-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
                    </svg>
                    <span>Yeni Sohbet</span>
                </button>
            </div>
            
            <div class="flex-1 overflow-y-auto" id="historyContainer">
                <div class="px-4" id="chatHistory">
                    <div class="text-xs text-gray-500 px-3 py-2 font-medium uppercase tracking-wider">Sohbet Geçmişi</div>
                    <div class="space-y-2" id="historyList">
                    </div>
                </div>
            </div>
            
            <div class="p-4 border-t border-gray-700/50">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 bg-gradient-to-br from-gray-600 to-gray-700 rounded-full flex items-center justify-center">
                        <span class="text-sm font-bold">K</span>
                    </div>
                    <div class="flex-1">
                        <div class="text-sm font-medium">Kullanıcı</div>
                        <div class="text-xs text-gray-400">localhost:5000</div>
                    </div>
                </div>
            </div>
        </aside>

        <main class="flex-1 flex flex-col bg-chat-bg">
            <header class="glass-effect p-4 border-b border-gray-700/50">
                <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-4">
                        <h2 class="text-xl font-semibold">G-Machine</h2>
                        <div class="flex items-center space-x-2">
                            <div class="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
                            <span class="text-xs text-yellow-400 font-medium">Yakın Zamanda Hizmette</span>
                        </div>
                    </div>
                    <div class="flex items-center space-x-3">
                        <button class="p-2 hover:bg-gray-700/50 rounded-lg transition-all duration-200" onclick="clearChat()">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                            </svg>
                        </button>
                    </div>
                </div>
            </header>

            <div class="flex-1 overflow-y-auto p-6 space-y-6 chat-container" id="chatArea">
                <div class="flex items-start space-x-4 message-animation">
                    <div class="w-10 h-10 bg-gradient-to-br from-wildberry to-wildberry-light rounded-full flex items-center justify-center flex-shrink-0 pulse-glow">
                        <span class="text-sm font-bold">۝</span>
                    </div>
                    <div class="flex-1">
                        <div class="message-ai rounded-2xl p-5 max-w-4xl">
                            <div class="flex items-center space-x-2 mb-3">
                                <span class="font-semibold text-wildberry-light">G-Machine</span>
                                <span class="text-xs text-gray-500">şimdi</span>
                            </div>
                            <div class="text-gray-100 leading-relaxed">
                                <p class="mb-2">👋 Merhaba! Ben G-Machine AI. Şuanda Aktif Değilim. Yakın Zamanda Kendimi Otomatik Şekilde Eğiteceğim. Şuanlık Lütfen Beni Bilgisayarında Eğit.</p>
                            </div>
                            <div class="warning-box">
                                <div class="flex items-center space-x-2 text-yellow-400">
                                    <span class="text-sm">⚠️</span>
                                    <span class="text-xs font-medium">Önemli: Tüm cevaplar çeşitli kaynaklardan derlenmiştir. Doğruluğunu teyit ediniz.</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <footer class="glass-effect border-t border-gray-700/50 p-6">
                <form id="chatForm" class="max-w-4xl mx-auto">
                    <div class="flex items-end space-x-4">
                        <div class="flex-1 relative">
                            <textarea 
                                id="messageInput" 
                                placeholder="G-Machine'e mesaj gönderin..." 
                                class="w-full bg-message-bg border border-gray-600/50 rounded-2xl px-5 py-4 pr-14 text-white placeholder-gray-400 focus:outline-none focus:border-wildberry input-glow transition-all duration-300 resize-none max-h-32"
                                rows="1"
                                autocomplete="off"
                            ></textarea>
                            <button 
                                type="submit" 
                                class="absolute right-3 bottom-3 bg-gradient-to-r from-wildberry to-wildberry-light hover:from-wildberry-dark hover:to-wildberry p-2.5 rounded-xl transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
                                id="sendButton"
                            >
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                                </svg>
                            </button>
                        </div>
                    </div>
                    <div class="text-xs text-gray-500 mt-3 text-center">
                        © 2025 Gexnys Her hakkı saklıdır.
                    </div>
                </form>
            </footer>
        </main>
    </div>

    <script>
        const chatArea = document.getElementById('chatArea');
        const messageInput = document.getElementById('messageInput');
        const chatForm = document.getElementById('chatForm');
        const sendButton = document.getElementById('sendButton');
        const newChatBtn = document.getElementById('newChatBtn');
        const historyList = document.getElementById('historyList');
        let currentChatId = 'chat-' + Date.now();
        let isProcessing = false;

        function initializeChatHistory() {
            const history = JSON.parse(localStorage.getItem('gMachineChatHistory') || '[]');
            historyList.innerHTML = '';
            
            history.slice(-10).reverse().forEach(item => {
                const historyItem = document.createElement('div');
                historyItem.className = 'sidebar-item px-3 py-3 cursor-pointer';
                historyItem.innerHTML = `
                    <div class="text-sm text-gray-300 font-medium truncate">${item.title}</div>
                    <div class="text-xs text-gray-500 mt-1">${item.time}</div>
                `;
                historyItem.onclick = () => loadChat(item.id);
                historyList.appendChild(historyItem);
            });
        }

        function saveToHistory(title, message) {
            const history = JSON.parse(localStorage.getItem('gMachineChatHistory') || '[]');
            const chatItem = {
                id: currentChatId,
                title: title || message.substring(0, 30) + '...',
                time: new Date().toLocaleTimeString('tr-TR'),
                messages: []
            };
            
            history.push(chatItem);
            if (history.length > 50) history.shift();
            localStorage.setItem('gMachineChatHistory', JSON.stringify(history));
            initializeChatHistory();
        }

        function loadChat(chatId) {
            currentChatId = chatId;
            clearChat();
        }

        function clearChat() {
            chatArea.innerHTML = `
                <div class="flex items-start space-x-4 message-animation">
                    <div class="w-10 h-10 bg-gradient-to-br from-wildberry to-wildberry-light rounded-full flex items-center justify-center flex-shrink-0 pulse-glow">
                        <span class="text-sm font-bold">۝</span>
                    </div>
                    <div class="flex-1">
                        <div class="message-ai rounded-2xl p-5 max-w-4xl">
                            <div class="flex items-center space-x-2 mb-3">
                                <span class="font-semibold text-wildberry-light">G-Machine</span>
                                <span class="text-xs text-gray-500">şimdi</span>
                            </div>
                            <div class="text-gray-100 leading-relaxed">
                                <p class="mb-2">👋 Merhaba! Ben G-Machine AI. Şuanda Aktif Değilim. Yakın Zamanda Kendimi Otomatik Şekilde Eğiteceğim. Şuanlık Lütfen Beni Bilgisayarında Eğit.</p>
                            </div>
                            <div class="warning-box">
                                <div class="flex items-center space-x-2 text-yellow-400">
                                    <span class="text-sm">⚠️</span>
                                    <span class="text-xs font-medium">Önemli: Tüm cevaplar çeşitli kaynaklardan derlenmiştir. Doğruluğunu teyit ediniz.</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            currentChatId = 'chat-' + Date.now();
        }

        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 128) + 'px';
        });

        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `flex items-start space-x-4 message-animation`;
            
            const timestamp = new Date().toLocaleTimeString('tr-TR', { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            
            if (isUser) {
                messageDiv.innerHTML = `
                    <div class="flex-1 flex justify-end">
                        <div class="message-user rounded-2xl p-4 max-w-3xl">
                            <div class="flex items-center justify-end space-x-2 mb-2">
                                <span class="text-xs text-gray-200">${timestamp}</span>
                                <span class="font-semibold text-white">Sen</span>
                            </div>
                            <div class="text-white leading-relaxed">${content}</div>
                        </div>
                    </div>
                    <div class="w-10 h-10 bg-gradient-to-br from-gray-600 to-gray-700 rounded-full flex items-center justify-center flex-shrink-0">
                        <span class="text-sm font-bold">K</span>
                    </div>
                `;
            } else {
                messageDiv.innerHTML = `
                    <div class="w-10 h-10 bg-gradient-to-br from-wildberry to-wildberry-light rounded-full flex items-center justify-center flex-shrink-0">
                        <span class="text-sm font-bold">G</span>
                    </div>
                    <div class="flex-1">
                        <div class="message-ai rounded-2xl p-5 max-w-4xl">
                            <div class="flex items-center space-x-2 mb-3">
                                <span class="font-semibold text-wildberry-light">G-Machine</span>
                                <span class="text-xs text-gray-500">${timestamp}</span>
                            </div>
                            <div class="text-gray-100 leading-relaxed">${formatMessage(content)}</div>
                            <div class="warning-box">
                                <div class="flex items-center space-x-2 text-yellow-400">
                                    <span class="text-sm">⚠️</span>
                                    <span class="text-xs font-medium">Önemli: Tüm cevaplar çeşitli kaynaklardan derlenmiştir. Doğruluğunu teyit ediniz.</span>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
            
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function formatMessage(content) {
            content = content.replace(/```python\n([\s\S]*?)```/g, '<div class="code-block"><code class="text-green-400">$1</code></div>');
            content = content.replace(/```html\n([\s\S]*?)```/g, '<div class="code-block"><code class="text-blue-400">$1</code></div>');
            content = content.replace(/```([\s\S]*?)```/g, '<div class="code-block"><code class="text-gray-300">$1</code></div>');
            content = content.replace(/\*\*(.*?)\*\*/g, '<strong class="text-wildberry-light">$1</strong>');
            content = content.replace(/\n/g, '<br>');
            content = content.replace(/🚀/g, '🚀 ');
            content = content.replace(/📚/g, '📚 ');
            content = content.replace(/⚠️/g, '⚠️ ');
            content = content.replace(/🐍/g, '🐍 ');
            content = content.replace(/🌐/g, '🌐 ');
            content = content.replace(/📊/g, '📊 ');
            content = content.replace(/🔍/g, '🔍 ');
            
            return content;
        }

        chatForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (isProcessing) {
                return;
            }
            
            const message = messageInput.value.trim();
            if (!message) return;

            console.log('Mesaj gönderiliyor:', message);
            isProcessing = true;
            sendButton.disabled = true;
            
            addMessage(message, true);
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            if (chatArea.querySelectorAll('.flex.items-start.space-x-4').length === 2) {
                saveToHistory('Yeni Sohbet', message);
            }

            const typingIndicator = document.createElement('div');
            typingIndicator.className = 'flex items-start space-x-4 message-animation';
            typingIndicator.innerHTML = `
                <div class="w-10 h-10 bg-gradient-to-br from-wildberry to-wildberry-light rounded-full flex items-center justify-center flex-shrink-0">
                    <span class="text-sm font-bold">G</span>
                </div>
                <div class="flex-1">
                    <div class="message-ai rounded-2xl p-5 max-w-4xl">
                        <div class="flex items-center space-x-2 mb-3">
                            <span class="font-semibold text-wildberry-light">G-Machine</span>
                            <span class="text-xs text-gray-500">yazıyor...</span>
                        </div>
                        <div class="text-gray-100 leading-relaxed">
                            <div class="flex space-x-1">
                                <div class="w-2 h-2 bg-wildberry rounded-full typing-dots"></div>
                                <div class="w-2 h-2 bg-wildberry rounded-full typing-dots" style="animation-delay: 0.2s"></div>
                                <div class="w-2 h-2 bg-wildberry rounded-full typing-dots" style="animation-delay: 0.4s"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            chatArea.appendChild(typingIndicator);
            chatArea.scrollTop = chatArea.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                console.log('AI yanıtı alındı:', data);
                
                typingIndicator.remove();
                
                if (data.response) {
                    addMessage(data.response, false);
                } else {
                    addMessage('❌ Yanıt alınamadı. Lütfen tekrar deneyin.', false);
                }
            } catch (error) {
                console.error('Hata:', error);
                typingIndicator.remove();
                addMessage('❌ G-Machine AI geçici olarak hata verdi. Lütfen tekrar deneyin.', false);
            } finally {
                isProcessing = false;
                sendButton.disabled = false;
                messageInput.focus();
            }
        });

        newChatBtn.addEventListener('click', function() {
            clearChat();
            currentChatId = 'chat-' + Date.now();
        });

        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                chatForm.dispatchEvent(new Event('submit'));
            }
        });

        window.addEventListener('load', function() {
            initializeChatHistory();
            messageInput.focus();
            console.log('🚀 G-Machine AI arayüzü hazır!');
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'response': '❌ Geçersiz veri formatı'}), 400
            
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'response': '❌ Lütfen bir mesaj girin'}), 400
        
        logger.info(f"Kullanıcı sorusu: {user_message}")
        
        ai_response = gmachine_ai.process_query(user_message)
        
        logger.info(f"AI yanıtı hazır: {len(ai_response)} karakter")
        
        return jsonify({'response': ai_response})
        
    except Exception as e:
        logger.error(f"Sohbet hatası: {str(e)}")
        return jsonify({'response': f'❌ Sistem hatası: {str(e)}'}), 500

@app.route('/history')
def get_history():
    return jsonify(gmachine_ai.conversation_history)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(gmachine_ai.conversation_history)
    })

def train_gmachine_model():
    print("🤖 G-Machine AI modeli eğitilemiyor...❌")
    try:
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        y = y.astype(int)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AdvancedViT().to(device)
        
        print("⚠️")
        print("✅")
        return model
        
    except Exception as e:
        print(f"❌ Model eğitim hatası: {e}")
        return None

if __name__ == '__main__':
    training_thread = threading.Thread(target=train_gmachine_model)
    training_thread.daemon = True
    training_thread.start()
    
    print("🚀 G-Machine AI Sistemi başlatılıyor...")
    print("🌐 Web arayüzü: http://localhost:5000 ✅")
    print("🎨 Wildberry teması aktif! ✅")
    print("🔍 Google araştırmalı AI hazır! ✅")
    print("✅ Sistem tamamen hazır! Mesaj göndermeyi deneyin.")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
     
    # Bu kod Gexnys tarafından yazılmıştır.
    # Tüm hakları Gexnys'e aittir.
    # Eklenmesi gereken kodlar için lütfen Gexnys ile iletişime geçin.
    # E-posta adresi : developergokhan@proton.me
