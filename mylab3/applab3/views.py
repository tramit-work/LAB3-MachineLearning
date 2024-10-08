from django.shortcuts import render

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
import re

def loadCsv(filename: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(filename)
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

def splitTrainTest(data, ratio_test):
    np.random.seed(28)
    index_permu = np.random.permutation(len(data))
    data_permu = data.iloc[index_permu]
    len_test = int(len(data_permu) * ratio_test)
    test_set = data_permu.iloc[:len_test, :]
    train_set = data_permu.iloc[len_test:, :]
    return train_set['Text'], train_set['Label'], test_set['Text'], test_set['Label']

def get_words_frequency(data_X):
    bag_words = np.unique(np.concatenate([text.split(' ') for text in data_X]))
    matrix_freq = np.zeros((len(data_X), len(bag_words)), dtype=int)
    word_freq = pd.DataFrame(matrix_freq, columns=bag_words)
    
    for id, text in enumerate(data_X):
        for word in text.split(' '):
            if word in bag_words:
                word_freq.at[id, word] += 1
    return word_freq, bag_words

def transform(data_test, bags):
    matrix_0 = np.zeros((len(data_test), len(bags)), dtype=int)
    frame_0 = pd.DataFrame(matrix_0, columns=bags)
    
    for id, text in enumerate(data_test):
        for word in text.split(' '):
            if word in bags:
                frame_0.at[id, word] += 1
    return frame_0

def cosine_distance(train_X, test_X):
    result = {}
    for i, test_row in enumerate(test_X):
        distances = []
        test_norm = np.linalg.norm(test_row)
        for j, train_row in enumerate(train_X):
            train_norm = np.linalg.norm(train_row)
            dot_product = np.dot(test_row, train_row)
            cosine_sim = dot_product / (test_norm * train_norm) if train_norm * test_norm != 0 else 0
            distances.append((cosine_sim, j))
        result[i] = sorted(distances, reverse=True)
    return result

class KNNText:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        distances = cosine_distance(self.X_train.values, X_test.values)
        predictions = {}
        
        for i in distances:
            nearest_neighbors = [self.y_train.iloc[idx] for _, idx in distances[i][:self.k]]
            predictions[i] = max(set(nearest_neighbors), key=nearest_neighbors.count)
        return predictions

def home(request):
    form = """
    <h1>Chào mừng đến với ứng dụng phân tích tâm trạng!</h1>
    <form method="post" action="/analyze_sentiment/">
        <textarea name="text" rows="4" cols="50" placeholder="Nhập câu của bạn ở đây..."></textarea><br>
        <input type="submit" value="Phân tích tâm trạng">
    </form>
    """
    return HttpResponse(form)

@csrf_exempt
def analyze_sentiment(request):
    if request.method == "POST":
        text = request.POST.get('text', '').strip()
        
        if not text:
            return HttpResponse("<h2>Vui lòng nhập văn bản để phân tích.</h2>")

        data = loadCsv("/Users/nguyenngocbaotram/Documents/HM&UD/LAB3-MachineLearning/mylab3/Data/Education.csv")
        if data.empty:
            return HttpResponse("<h2>Không thể tải dữ liệu. Vui lòng kiểm tra file CSV.</h2>")
        
        data['Text'] = data['Text'].apply(lambda x: re.sub(r'[,.]', '', x))
        X_train, y_train, _, _ = splitTrainTest(data, 0.25)

        words_train_fre, bags = get_words_frequency(X_train)
        text_cleaned = re.sub(r'[,.]', '', text)
        words_test_fre = transform(pd.Series([text_cleaned]), bags)

        knn = KNNText(k=2)
        knn.fit(words_train_fre, y_train)

        prediction = knn.predict(words_test_fre)

        pred_label = list(prediction.values())[0]

        result = f"""
        <h2>Kết quả phân tích tâm trạng cho văn bản: '{text}'</h2>
        <h3>Dự đoán của KNN: {pred_label}</h3>
        <a href='/'>Quay lại</a>
        """
        return HttpResponse(result)
    else:
        return HttpResponse("<h2>Chỉ chấp nhận phương thức POST.</h2>")
