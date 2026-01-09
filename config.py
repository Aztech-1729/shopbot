# =========================
# BOT CONFIGURATION
# =========================
import os

BOT_TOKEN = "8461858373:AAGi5rGKKlcRu-n0LNMW7Nx0SVHteJtz_vw"
BOT_USERNAME = "Optic_Shop_bot"

# Force join (private invite link supported)
FORCE_JOIN_CHANNEL = "https://t.me/+4nRUAm6UdUUwYWQ9"

START_IMAGE = "https://i.postimg.cc/z8VTfHV1/start.jpg"

# Support username (without @)
SUPPORT_USERNAME = "Soulxmerchant"

# =========================
# ADMIN SETTINGS
# =========================
ADMIN_IDS = [
    6670166083,
]

# =========================
# MongoDB
# =========================
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://aztech:ayazahmed1122@cluster0.mhuaw3q.mongodb.net/clientott_db?retryWrites=true&w=majority",
)
DB_NAME = os.getenv("DB_NAME", "clientott_db")

# =========================
# PAYMENT CONFIG
# =========================
PAYMENTS = {
    "binance": {
        "name": "Binance Pay",
        "qr_image": "https://i.postimg.cc/Gt9sPyxP/binance.jpg",
        "binance_id": "1131270499",
        "note": "Send via Binance UID Transfer",
    },
    "upi": {
        "name": "UPI",
        "qr_image": "https://i.postimg.cc/sxp144Z7/upi.jpg",
        "upi_id": "soulxchandles@fam",
        "upi_name": "AKASH CHAND",
    },
}

# =========================
# Currency conversion
# =========================
# Used only for displaying wallet balance in USD when user selects USD.
USD_INR_RATE = 89.86

# =========================
# Product images (used in Buy Products screens)
# =========================
# The bot will pick an image based on keywords in product name.
PRODUCT_IMAGE_RULES = [
    ("adobe", "https://www.dolphincomputer.co.in/wp-content/uploads/2024/05/3-1080x599.png"),
    ("chatgpt", "https://www.internetmatters.org/wp-content/uploads/2025/06/Chat-GPT-logo.webp"),
    ("gemini", "https://www.gstatic.com/lamda/images/gemini_aurora_thumbnail_4g_e74822ff0ca4259beb718.png"),
    ("github student", "https://muralisugumar.com/wp-content/uploads/2024/11/maxresdefault.jpg"),
    ("perplexity", "https://cdn.analyticsvidhya.com/wp-content/uploads/2025/08/Everything-you-need-to-know-about-Perplexity-Pro.webp"),
    ("amazon blank", "https://m.media-amazon.com/images/I/31epF-8N9LL.png"),
    ("amazon prime", "https://images.moneycontrol.com/static-mcnews/2024/07/20240725074952_WhatsApp-Image-2024-07-25-at-12.17.26.jpeg"),
    ("spotify", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZHkw8hJp2hf8jYZ3fvIgTyOvsEJ1xo8Hefw&s"),
    ("youtube", "https://www.gstatic.com/youtube/img/promos/growth/YTP_logo_social_1200x630.png?days_since_epoch=20461"),
    ("nordvpn", "https://static0.howtogeekimages.com/wordpress/wp-content/uploads/2025/01/nordvpn-3.jpeg"),
    ("surfshark", "https://www.thefastmode.com/media/k2/items/src/62ac1f5c67d50ada30f1e7759102fb13.jpg?t=20230802_100215"),
]
