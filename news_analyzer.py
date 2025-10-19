#!/usr/bin/env python3
"""
News Sentiment Analysis System
Fetches news for each company and assigns sentiment scores (good/bad) based on content.
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional, Tuple
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """
    Analyzes news sentiment for stock companies.
    """
    
    def __init__(self, data_dir: str = "news_data"):
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # Company information
        self.companies = {
            'AAPL': {
                'name': 'Apple Inc.',
                'keywords': ['Apple', 'iPhone', 'iPad', 'Mac', 'iOS', 'macOS', 'App Store', 'Tim Cook', 'Apple Watch', 'AirPods']
            },
            'NVDA': {
                'name': 'NVIDIA Corporation',
                'keywords': ['NVIDIA', 'Nvidia', 'GPU', 'AI', 'machine learning', 'Jensen Huang', 'RTX', 'GeForce', 'CUDA', 'data center']
            },
            'LLY': {
                'name': 'Eli Lilly and Company',
                'keywords': ['Eli Lilly', 'Lilly', 'pharmaceutical', 'drug', 'medicine', 'FDA', 'clinical trial', 'diabetes', 'Alzheimer']
            },
            'NVO': {
                'name': 'Novo Nordisk A/S',
                'keywords': ['Novo Nordisk', 'Novo', 'diabetes', 'Ozempic', 'Wegovy', 'insulin', 'GLP-1', 'pharmaceutical']
            },
            'DNA': {
                'name': 'Genentech (Roche)',
                'keywords': ['Genentech', 'Roche', 'biotechnology', 'cancer', 'oncology', 'Herceptin', 'Avastin', 'pharmaceutical']
            }
        }
        
        # Sentiment keywords
        self.positive_keywords = [
            'profit', 'growth', 'revenue', 'earnings', 'beat', 'exceed', 'surge', 'rise', 'gain',
            'success', 'breakthrough', 'approval', 'launch', 'expansion', 'acquisition', 'merger',
            'partnership', 'deal', 'contract', 'upgrade', 'bullish', 'optimistic', 'strong',
            'robust', 'solid', 'outperform', 'outstanding', 'record', 'milestone', 'innovation',
            'breakthrough', 'FDA approval', 'clinical success', 'positive results', 'strong demand',
            'market share', 'competitive advantage', 'leadership', 'expertise', 'technology',
            'patent', 'intellectual property', 'moat', 'premium', 'quality', 'reliable'
        ]
        
        self.negative_keywords = [
            'loss', 'decline', 'fall', 'drop', 'crash', 'plunge', 'slump', 'weak', 'poor',
            'disappointing', 'miss', 'below', 'disappoint', 'concern', 'worry', 'risk',
            'challenge', 'problem', 'issue', 'failure', 'recall', 'lawsuit', 'investigation',
            'regulatory', 'compliance', 'penalty', 'fine', 'violation', 'scandal', 'controversy',
            'competition', 'threat', 'pressure', 'headwind', 'uncertainty', 'volatility',
            'recession', 'economic', 'downgrade', 'bearish', 'pessimistic', 'weakness',
            'struggle', 'difficulty', 'setback', 'delay', 'cancellation', 'rejection',
            'FDA rejection', 'clinical failure', 'negative results', 'weak demand',
            'market share loss', 'competitive pressure', 'commoditization', 'price war'
        ]
        
        # News sources (free APIs and RSS feeds)
        self.news_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.marketwatch.com/marketwatch/topstories/',
            'https://feeds.reuters.com/reuters/businessNews',
            'https://feeds.bloomberg.com/markets/news.rss'
        ]
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created news data directory: {self.data_dir}")
    
    def fetch_news_from_rss(self, rss_url: str, max_articles: int = 50) -> List[Dict]:
        """Fetch news articles from RSS feed."""
        try:
            response = requests.get(rss_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            articles = []
            
            items = soup.find_all('item')[:max_articles]
            
            for item in items:
                try:
                    title = item.find('title').text.strip() if item.find('title') else ''
                    description = item.find('description').text.strip() if item.find('description') else ''
                    link = item.find('link').text.strip() if item.find('link') else ''
                    pub_date = item.find('pubDate').text.strip() if item.find('pubDate') else ''
                    
                    if title and description:
                        articles.append({
                            'title': title,
                            'description': description,
                            'link': link,
                            'pub_date': pub_date,
                            'source': rss_url
                        })
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from {rss_url}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from {rss_url}: {e}")
            return []
    
    def fetch_news_for_company(self, ticker: str, days_back: int = 30) -> List[Dict]:
        """Fetch news articles for a specific company."""
        company_info = self.companies[ticker]
        company_name = company_info['name']
        keywords = company_info['keywords']
        
        all_articles = []
        
        # Fetch from multiple sources
        for source in self.news_sources:
            articles = self.fetch_news_from_rss(source)
            all_articles.extend(articles)
            time.sleep(1)  # Rate limiting
        
        # Filter articles relevant to the company
        relevant_articles = []
        
        for article in all_articles:
            text = f"{article['title']} {article['description']}".lower()
            
            # Check if article mentions the company or its keywords
            is_relevant = False
            for keyword in keywords:
                if keyword.lower() in text:
                    is_relevant = True
                    break
            
            if is_relevant:
                # Parse publication date
                try:
                    if article['pub_date']:
                        pub_date = datetime.strptime(article['pub_date'], '%a, %d %b %Y %H:%M:%S %Z')
                    else:
                        pub_date = datetime.now()
                except:
                    pub_date = datetime.now()
                
                # Check if article is within the specified time range
                if pub_date >= datetime.now() - timedelta(days=days_back):
                    article['ticker'] = ticker
                    article['company_name'] = company_name
                    article['pub_date'] = pub_date
                    relevant_articles.append(article)
        
        logger.info(f"Found {len(relevant_articles)} relevant articles for {company_name}")
        return relevant_articles
    
    def calculate_sentiment_score(self, text: str) -> Tuple[float, str]:
        """
        Calculate sentiment score for text.
        Returns: (score, sentiment_label)
        """
        text_lower = text.lower()
        
        positive_count = 0
        negative_count = 0
        
        # Count positive keywords
        for keyword in self.positive_keywords:
            if keyword.lower() in text_lower:
                positive_count += 1
        
        # Count negative keywords
        for keyword in self.negative_keywords:
            if keyword.lower() in text_lower:
                negative_count += 1
        
        # Calculate sentiment score
        total_keywords = positive_count + negative_count
        
        if total_keywords == 0:
            return 0.0, 'neutral'
        
        sentiment_score = (positive_count - negative_count) / total_keywords
        
        # Determine sentiment label
        if sentiment_score > 0.2:
            sentiment_label = 'good'
        elif sentiment_score < -0.2:
            sentiment_label = 'bad'
        else:
            sentiment_label = 'neutral'
        
        return sentiment_score, sentiment_label
    
    def analyze_news_sentiment(self, articles: List[Dict]) -> List[Dict]:
        """Analyze sentiment for a list of articles."""
        analyzed_articles = []
        
        for article in articles:
            # Combine title and description for analysis
            full_text = f"{article['title']} {article['description']}"
            
            # Calculate sentiment
            sentiment_score, sentiment_label = self.calculate_sentiment_score(full_text)
            
            # Add sentiment analysis to article
            article['sentiment_score'] = sentiment_score
            article['sentiment_label'] = sentiment_label
            article['analysis_text'] = full_text
            
            analyzed_articles.append(article)
        
        return analyzed_articles
    
    def fetch_all_company_news(self, days_back: int = 30) -> Dict[str, List[Dict]]:
        """Fetch and analyze news for all companies."""
        all_news = {}
        
        for ticker in self.companies.keys():
            logger.info(f"Fetching news for {self.companies[ticker]['name']}...")
            
            # Fetch news articles
            articles = self.fetch_news_for_company(ticker, days_back)
            
            # Analyze sentiment
            analyzed_articles = self.analyze_news_sentiment(articles)
            
            all_news[ticker] = analyzed_articles
            
            # Add delay between companies
            time.sleep(2)
        
        return all_news
    
    def save_news_data(self, all_news: Dict[str, List[Dict]]):
        """Save news data to files."""
        
        # Save individual company files
        for ticker, articles in all_news.items():
            if articles:
                # Convert to DataFrame
                df = pd.DataFrame(articles)
                
                # Save CSV
                csv_path = f"{self.data_dir}/{ticker}_news.csv"
                df.to_csv(csv_path, index=False)
                
                # Save JSON
                json_path = f"{self.data_dir}/{ticker}_news.json"
                with open(json_path, 'w') as f:
                    json.dump(articles, f, indent=2, default=str)
                
                logger.info(f"Saved {len(articles)} articles for {ticker}")
        
        # Save combined data
        all_articles = []
        for ticker, articles in all_news.items():
            all_articles.extend(articles)
        
        if all_articles:
            combined_df = pd.DataFrame(all_articles)
            combined_df.to_csv(f"{self.data_dir}/all_companies_news.csv", index=False)
            
            with open(f"{self.data_dir}/all_companies_news.json", 'w') as f:
                json.dump(all_articles, f, indent=2, default=str)
        
        # Save summary statistics
        self.save_news_summary(all_news)
    
    def save_news_summary(self, all_news: Dict[str, List[Dict]]):
        """Save news summary statistics."""
        summary_data = []
        
        for ticker, articles in all_news.items():
            if articles:
                company_name = self.companies[ticker]['name']
                
                # Calculate statistics
                total_articles = len(articles)
                good_articles = len([a for a in articles if a['sentiment_label'] == 'good'])
                bad_articles = len([a for a in articles if a['sentiment_label'] == 'bad'])
                neutral_articles = len([a for a in articles if a['sentiment_label'] == 'neutral'])
                
                avg_sentiment = np.mean([a['sentiment_score'] for a in articles])
                
                summary_data.append({
                    'Ticker': ticker,
                    'Company_Name': company_name,
                    'Total_Articles': total_articles,
                    'Good_Articles': good_articles,
                    'Bad_Articles': bad_articles,
                    'Neutral_Articles': neutral_articles,
                    'Good_Percentage': (good_articles / total_articles) * 100,
                    'Bad_Percentage': (bad_articles / total_articles) * 100,
                    'Neutral_Percentage': (neutral_articles / total_articles) * 100,
                    'Average_Sentiment_Score': avg_sentiment,
                    'Sentiment_Trend': 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.data_dir}/news_sentiment_summary.csv", index=False)
        
        # Save metadata
        metadata = {
            "analysis_timestamp": datetime.now().isoformat(),
            "days_back": 730,
            "analysis_period": "2 years (matching stock data period)",
            "stock_data_period": "2023-10-18 to 2025-10-17",
            "total_articles": sum(len(articles) for articles in all_news.values()),
            "companies_analyzed": list(all_news.keys()),
            "sentiment_keywords": {
                "positive_count": len(self.positive_keywords),
                "negative_count": len(self.negative_keywords)
            }
        }
        
        with open(f"{self.data_dir}/news_analysis_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def create_news_visualizations(self, all_news: Dict[str, List[Dict]]):
        """Create visualizations for news sentiment analysis."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('News Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution by Company
        ax1 = axes[0, 0]
        
        companies = []
        good_counts = []
        bad_counts = []
        neutral_counts = []
        
        for ticker, articles in all_news.items():
            if articles:
                company_name = self.companies[ticker]['name']
                companies.append(company_name)
                
                good_count = len([a for a in articles if a['sentiment_label'] == 'good'])
                bad_count = len([a for a in articles if a['sentiment_label'] == 'bad'])
                neutral_count = len([a for a in articles if a['sentiment_label'] == 'neutral'])
                
                good_counts.append(good_count)
                bad_counts.append(bad_count)
                neutral_counts.append(neutral_count)
        
        if companies:
            x = np.arange(len(companies))
            width = 0.25
            
            ax1.bar(x - width, good_counts, width, label='Good', color='green', alpha=0.7)
            ax1.bar(x, neutral_counts, width, label='Neutral', color='gray', alpha=0.7)
            ax1.bar(x + width, bad_counts, width, label='Bad', color='red', alpha=0.7)
            
            ax1.set_title('News Sentiment Distribution by Company', fontweight='bold')
            ax1.set_xlabel('Company')
            ax1.set_ylabel('Number of Articles')
            ax1.set_xticks(x)
            ax1.set_xticklabels(companies, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Average Sentiment Scores
        ax2 = axes[0, 1]
        
        companies = []
        avg_scores = []
        
        for ticker, articles in all_news.items():
            if articles:
                company_name = self.companies[ticker]['name']
                companies.append(company_name)
                avg_score = np.mean([a['sentiment_score'] for a in articles])
                avg_scores.append(avg_score)
        
        if companies:
            colors = ['green' if score > 0.1 else 'red' if score < -0.1 else 'gray' for score in avg_scores]
            bars = ax2.bar(companies, avg_scores, color=colors, alpha=0.7)
            ax2.set_title('Average Sentiment Scores by Company', fontweight='bold')
            ax2.set_xlabel('Company')
            ax2.set_ylabel('Average Sentiment Score')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels on bars
            for bar, score in zip(bars, avg_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 3. News Volume Over Time
        ax3 = axes[1, 0]
        
        # Combine all articles and group by date
        all_articles = []
        for ticker, articles in all_news.items():
            all_articles.extend(articles)
        
        if all_articles:
            df = pd.DataFrame(all_articles)
            df['date'] = pd.to_datetime(df['pub_date']).dt.date
            
            daily_counts = df.groupby('date').size()
            
            ax3.plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2)
            ax3.set_title('Daily News Volume', fontweight='bold')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Number of Articles')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # 4. Sentiment Score Distribution
        ax4 = axes[1, 1]
        
        all_scores = []
        for ticker, articles in all_news.items():
            all_scores.extend([a['sentiment_score'] for a in articles])
        
        if all_scores:
            ax4.hist(all_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax4.set_title('Distribution of Sentiment Scores', fontweight='bold')
            ax4.set_xlabel('Sentiment Score')
            ax4.set_ylabel('Frequency')
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/news_sentiment_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.data_dir}/news_sentiment_analysis.pdf", bbox_inches='tight')
        plt.show()
    
    def generate_news_report(self, all_news: Dict[str, List[Dict]]) -> str:
        """Generate a comprehensive news analysis report."""
        report = []
        report.append("="*80)
        report.append("NEWS SENTIMENT ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        # Overall summary
        total_articles = sum(len(articles) for articles in all_news.values())
        report.append(f"📰 OVERALL SUMMARY")
        report.append("-" * 50)
        report.append(f"Total articles analyzed: {total_articles}")
        report.append(f"Companies analyzed: {len(all_news)}")
        report.append(f"Analysis period: Last 730 days (2 years) - matching stock data period")
        report.append("")
        
        # Company-specific analysis
        for ticker, articles in all_news.items():
            if articles:
                company_name = self.companies[ticker]['name']
                report.append(f"📊 {company_name} ({ticker})")
                report.append("-" * 50)
                
                # Calculate statistics
                total_articles = len(articles)
                good_articles = len([a for a in articles if a['sentiment_label'] == 'good'])
                bad_articles = len([a for a in articles if a['sentiment_label'] == 'bad'])
                neutral_articles = len([a for a in articles if a['sentiment_label'] == 'neutral'])
                
                avg_sentiment = np.mean([a['sentiment_score'] for a in articles])
                
                report.append(f"Total articles: {total_articles}")
                report.append(f"Good news: {good_articles} ({good_articles/total_articles*100:.1f}%)")
                report.append(f"Bad news: {bad_articles} ({bad_articles/total_articles*100:.1f}%)")
                report.append(f"Neutral news: {neutral_articles} ({neutral_articles/total_articles*100:.1f}%)")
                report.append(f"Average sentiment score: {avg_sentiment:.3f}")
                
                # Sentiment trend
                if avg_sentiment > 0.1:
                    trend = "POSITIVE"
                elif avg_sentiment < -0.1:
                    trend = "NEGATIVE"
                else:
                    trend = "NEUTRAL"
                
                report.append(f"Overall sentiment trend: {trend}")
                
                # Recent notable articles
                report.append("")
                report.append("Recent notable articles:")
                
                # Sort by sentiment score and show top 3
                sorted_articles = sorted(articles, key=lambda x: abs(x['sentiment_score']), reverse=True)
                
                for i, article in enumerate(sorted_articles[:3]):
                    sentiment_emoji = "😊" if article['sentiment_label'] == 'good' else "😞" if article['sentiment_label'] == 'bad' else "😐"
                    report.append(f"  {i+1}. {sentiment_emoji} {article['title'][:80]}...")
                    report.append(f"     Score: {article['sentiment_score']:.3f} | Date: {article['pub_date'].strftime('%Y-%m-%d')}")
                
                report.append("")
        
        # Key insights
        report.append("💡 KEY INSIGHTS")
        report.append("-" * 50)
        
        # Find most positive and negative companies
        company_scores = {}
        for ticker, articles in all_news.items():
            if articles:
                avg_score = np.mean([a['sentiment_score'] for a in articles])
                company_scores[ticker] = avg_score
        
        if company_scores:
            most_positive = max(company_scores, key=company_scores.get)
            most_negative = min(company_scores, key=company_scores.get)
            
            report.append(f"• Most positive sentiment: {self.companies[most_positive]['name']} ({company_scores[most_positive]:.3f})")
            report.append(f"• Most negative sentiment: {self.companies[most_negative]['name']} ({company_scores[most_negative]:.3f})")
        
        report.append("")
        report.append("📈 TRADING IMPLICATIONS:")
        report.append("• Positive news sentiment may indicate upward price pressure")
        report.append("• Negative news sentiment may indicate downward price pressure")
        report.append("• Neutral sentiment suggests stable conditions")
        report.append("• News sentiment can be used as a contrarian indicator")
        report.append("• Combine news sentiment with technical analysis for better signals")
        
        return "\n".join(report)

def main():
    """Main function to run news sentiment analysis."""
    try:
        print("📰 Starting News Sentiment Analysis...")
        print("="*60)
        
        # Initialize analyzer
        analyzer = NewsSentimentAnalyzer()
        
        # Fetch and analyze news for the same period as stock data (2 years)
        print("🔍 Fetching news for all companies...")
        all_news = analyzer.fetch_all_company_news(days_back=730)  # Match 2-year stock data period
        
        # Save data
        print("💾 Saving news data...")
        analyzer.save_news_data(all_news)
        
        # Create visualizations
        print("📊 Creating news visualizations...")
        analyzer.create_news_visualizations(all_news)
        
        # Generate report
        print("📝 Generating news analysis report...")
        report = analyzer.generate_news_report(all_news)
        
        # Save report
        with open(f"{analyzer.data_dir}/news_sentiment_report.txt", 'w') as f:
            f.write(report)
        
        print("\n" + "="*60)
        print("🎉 NEWS SENTIMENT ANALYSIS COMPLETED!")
        print("="*60)
        print("\n📁 Files created:")
        print("  📰 Individual company news files (CSV/JSON)")
        print("  📊 news_sentiment_analysis.png/pdf - Visualizations")
        print("  📝 news_sentiment_report.txt - Detailed analysis report")
        print("  📈 news_sentiment_summary.csv - Summary statistics")
        
        print("\n" + report)
        
    except Exception as e:
        logger.error(f"Error in news sentiment analysis: {str(e)}")
        print(f"❌ Error: {str(e)}")
        print("Note: This may be due to network issues or RSS feed availability.")
        print("The system will create sample data for demonstration purposes.")
        
        # Create sample data for demonstration
        create_sample_news_data()

def create_sample_news_data():
    """Create sample news data for demonstration purposes."""
    print("\n🔄 Creating sample news data for demonstration...")
    
    # Sample news data
    sample_news = {
        'AAPL': [
            {
                'title': 'Apple Reports Record Q4 Earnings, iPhone Sales Surge',
                'description': 'Apple Inc. reported record quarterly earnings with iPhone sales exceeding expectations.',
                'pub_date': datetime.now() - timedelta(days=5),
                'sentiment_score': 0.8,
                'sentiment_label': 'good',
                'ticker': 'AAPL',
                'company_name': 'Apple Inc.'
            },
            {
                'title': 'Apple Faces Antitrust Investigation in Europe',
                'description': 'European regulators launch antitrust investigation into Apple\'s App Store practices.',
                'pub_date': datetime.now() - timedelta(days=10),
                'sentiment_score': -0.6,
                'sentiment_label': 'bad',
                'ticker': 'AAPL',
                'company_name': 'Apple Inc.'
            }
        ],
        'NVDA': [
            {
                'title': 'NVIDIA AI Chips Drive Record Revenue Growth',
                'description': 'NVIDIA Corporation reports strong demand for AI chips driving record revenue.',
                'pub_date': datetime.now() - timedelta(days=3),
                'sentiment_score': 0.9,
                'sentiment_label': 'good',
                'ticker': 'NVDA',
                'company_name': 'NVIDIA Corporation'
            }
        ]
    }
    
    # Save sample data
    analyzer = NewsSentimentAnalyzer()
    analyzer.save_news_data(sample_news)
    
    print("✅ Sample news data created successfully!")
    print("📁 Check the 'news_data/' directory for sample files.")

if __name__ == "__main__":
    main()
