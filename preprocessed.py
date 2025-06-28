# preprocessed.py

import pandas as pd
import regex
from underthesea import sentiment, word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
import emoji
import nltk

# Khởi tạo danh sách từ tiêu cực, tích cực
positive_words = [
    "thích", "tốt", "xuất sắc", "tuyệt vời", "ổn","hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "nhanh",
    "thân thiện", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp","an tâm", "thúc đẩy", "cảm động", 
    "nổi trội","sáng tạo", "phù hợp", "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "hào hứng", "đam mê", 'chuyên', 'dễ', 'giỏi', 'hay', 'hiệu', 'hài', 'hỗ trợ', 'nhiệt tình',
    'thân', 'tuyệt', 'vui', 'chuyên nghiệp', 'động lực', 'dễ chịu','công bằng', 'hạnh phúc', 'hợp lý','truyền cảm hứng', 
    'phát triển', 'nổi bật','hợp tác', 'đồng đội', 'hòa đồng', 'học hỏi', 'tôn trọng', 'tốt nhất', 'vui mừng', 'đẳng cấp',
    'dễ dàng', 'chủ động', 'đồng cảm', 'cảm', 'mở rộng', 'bình đẳng', 'năng động', 'thoải mái', 'mến', 'cảm ơn', 'tốt hơn','cởi mở', 'cơ hội'
]
negative_words = [
    "kém", "tệ", "buồn", "không dễ chịu", "không thích", "không ổn", "áp lực", "mệt","không đáng tin cậy", "không chuyên nghiệp",
    "không thân thiện", "không tốt", "chậm", "khó khăn", "phức tạp",
    "khó chịu", "gây khó dễ", "rườm rà", "tồi tệ", "khó xử", "không thể chấp nhận", "không rõ ràng",
    "rối rắm", 'không hài lòng', 'quá tệ', 'rất tệ', "phiền phức",
    'thất vọng', 'tệ hại', 'kinh khủng', 'chán', 'drama', 'dramas', 'gáp', 'gắt',
    'lỗi', 'ngắt', 'quái', 'quát', 'thiếu', 'trễ', 'tệp', 'tồi', "hách dịch",
    'căng thẳng', 'không hòa đồng', 'thiếu đào tạo', 'thiếu sáng tạo',
    'khủng hoảng', 'rối loạn', 'không có cơ hội', 'thiếu công bằng', 'không chấp nhận được',
    'không đủ', 'thiếu sự công nhận', 'thiếu hỗ trợ', 'không hợp', 'thiếu cơ hội thăng tiến', 'áp', 'trì trệ', 'thất bại',
    'thiếu sự minh bạch', 'buồn bã', 'rối', 'không đáng', 'mâu thuẫn',
    'thiếu chuyên nghiệp', 'thiếu động lực', 'lo lắng', 'môi trường thiếu cởi mở', 'mệt mỏi','lo'
    'thiếu linh hoạt', 'không tôn trọng', 'tức giận', 'không phát triển', 'thiếu sự rõ ràng', 'bực bội'
]

features =[
    'Salary & benefits',
    'Training & learning',
    'Management cares about me',
    'Culture & fun',
    'Office & workspace'
]


# Danh sách từ dừng tiếng Việt
vietnamese_stopwords = [
    "là", "của", "và", "có", "được", "trong", "cho", "với", "tại", "bởi",
    "để", "mà", "này", "kia", "nó", "họ", "như", "thì", "đã", "đang",
    "rồi", "một", "những", "các", "từng", "cũng", "ra", "vào", "trên",
    "dưới", "gì", "khi", "nào", "đâu", "ai", "bằng", "theo", "về"
]

# Từ điển emoji
emoji_dict = {
    "😊": "vui",
    "😢": "buồn",
    "👍": "tích cực",
    "👎": "tiêu cực"
}

# Từ điển teencode
teen_dict = {
    "k": "không",
    "ko": "không",
    "ng": "người",
    "nt": "nhắn tin",
    "ok": "ổn",
    "oke": "ổn"
}

# Danh sách từ sai
wrong_lst = ["", " ", "\n"]

def process_text(text):
    """
    Tiền xử lý văn bản để chuẩn bị cho phân tích cảm xúc

    Parameters:
        text (str): đoạn văn bản đầu vào

    Returns:
        str: văn bản đã được xử lý
    """
    # Chuyển thành chữ thường và loại bỏ ký tự không cần thiết
    document = text.lower()
    document = document.replace("’", '')
    document = regex.sub(r'\.+', ".", document)

    new_sentence = ''
    for sentence in sent_tokenize(document):
        # Chuyển emoji thành từ
        sentence = ''.join(emoji_dict.get(word, word) + ' ' for word in list(sentence))
        # Chuyển teencode
        sentence = ' '.join(teen_dict.get(word, word) for word in sentence.split())
        # Loại bỏ từ sai
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        # Loại bỏ từ dừng tiếng Anh và tiếng Việt
        stop_words = set(stopwords.words('english')) | set(vietnamese_stopwords)
        sentence = ' '.join(word for word in sentence.split() if word not in stop_words)
        new_sentence += sentence + '. '
    
    # Chuẩn hóa khoảng trắng
    document = regex.sub(r'\s+', ' ', new_sentence).strip()
    return document

def count_pos_neg_words(text, pos_words=positive_words, neg_words=negative_words):
    """
    Đếm số câu/cụm từ tích cực và tiêu cực trong văn bản sử dụng underthesea và danh sách từ dự phòng

    Parameters:
        text (str): đoạn văn bản đầu vào
        pos_words (list): danh sách từ/cụm từ tích cực
        neg_words (list): danh sách từ/cụm từ tiêu cực

    Returns:
        pos_count (int), neg_count (int): số câu/cụm từ tích cực và tiêu cực
    """
    # Tiền xử lý văn bản
    text = process_text(text)
    
    # Tách câu hoặc cụm từ
    sentences = [s.strip() for s in text.replace(".", ". ").replace(",", ", ").split(". ") if s.strip()]
    
    pos_count = 0
    neg_count = 0
    
    # Phân tích cảm xúc từng câu/cụm từ bằng underthesea
    for sentence in sentences:
        tokenized_sentence = " ".join(word_tokenize(sentence))
        result = sentiment(tokenized_sentence)
        if result == "positive":
            pos_count += 1
        elif result == "negative":
            neg_count += 1
        else:
            # Dự phòng: Kiểm tra danh sách từ nếu underthesea không nhận diện được
            tokens = tokenized_sentence.split()
            for i in range(len(tokens) - 1):
                phrase = tokens[i] + " " + tokens[i + 1]
                if phrase in pos_words:
                    pos_count += 1
                    tokens[i] = ""
                    tokens[i + 1] = ""
                elif phrase in neg_words:
                    neg_count += 1
                    tokens[i] = ""
                    tokens[i + 1] = ""
            # Đếm từ đơn
            pos_count += sum(1 for token in tokens if token in pos_words and token)
            neg_count += sum(1 for token in tokens if token in neg_words and token)
    
    return pos_count, neg_count

def classify_sentiment(row):
    """
    Gán nhãn cảm xúc dựa vào điểm đánh giá và số câu/cụm từ cảm xúc

    Parameters:
        row (pd.Series): 1 dòng dữ liệu chứa Rating, word_count_positive, word_count_negative

    Returns:
        int: 0 = negative, 1 = neutral, 2 = positive
    """
    rating = row['Rating']
    pos_count = row['word_count_positive']
    neg_count = row['word_count_negative']
    
    if rating >= 4:
        return 2
    elif rating == 3:
        if pos_count > neg_count:
            return 2
        elif pos_count < neg_count:
            return 0
        else:
            return 1
    elif rating < 3:
        if rating == 0:
            if pos_count > neg_count:
                return 2
            elif pos_count < neg_count:
                return 0
            else:
                return 0
        else:
            return 0