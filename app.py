from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model và scaler
model = pickle.load(open("best_xgboost_model.pkl", "rb"))
scalerIn = pickle.load(open("scalerInput.pkl", "rb"))
scalerOut = pickle.load(open("scalerOutput.pkl", "rb"))

# Hàm dự đoán giá nhà
def pred(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, proximity):
    # Mapping proximity từ form vào giá trị encoding
    proximity_mapping = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    ocean_proximity = proximity_mapping[proximity]

    # Dummies encoding cho proximity
    ocean_mapping = {
        "<1H OCEAN": [1, 0, 0, 0, 0],
        "INLAND": [0, 1, 0, 0, 0],
        "NEAR OCEAN": [0, 0, 1, 0, 0],
        "NEAR BAY": [0, 0, 0, 1, 0],
        "ISLAND": [0, 0, 0, 0, 1],
    }
    ocean_dummies = ocean_mapping[ocean_proximity]

    # Tính toán thêm các feature
    rooms_per_household = total_rooms / households if households > 0 else 0 #Đại diện cho những khu vực nhà ở lớn, rộng rãi, nơi mỗi hộ gia đình có nhiều phòng.
    population_per_household = population / households if households > 0 else 0 #Biểu thị các khu vực đông đúc hoặc các hộ gia đình
    household_density = households / population if population > 0 else 0 #Biểu diễn mật độ dân số trong khu vực
    people_per_room = population / total_rooms if total_rooms > 0 else 0 #Số người trong 1 phòng

    # Tạo feature vector
    input_features = [
        longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
        population, households, median_income
    ] + ocean_dummies + [rooms_per_household, population_per_household, household_density, people_per_room]

    # Tạo DataFrame với đúng tên cột
    input_df = pd.DataFrame([input_features], columns=[
        "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
        "population", "households", "median_income", "1H OCEAN", "INLAND",
        "ISLAND", "NEAR BAY", "NEAR OCEAN", "rooms_per_household", "population_per_household",
        "household_density", "people_per_room"
    ])

    input_features_scaled = scalerIn.transform(input_df)

    prediction = model.predict(input_features_scaled)
    return scalerOut.inverse_transform(prediction[0].reshape(-1, 1))[0][0]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/house', methods=['POST'])
def house():
    try:
        # Lấy dữ liệu từ form
        longitude = float(request.form.get('longitude', 0))
        latitude = float(request.form.get('latitude', 0))
        age = float(request.form.get('age', 0))
        rooms = float(request.form.get('rooms', 0))
        bedrooms = float(request.form.get('bedrooms', 0))
        population = float(request.form.get('population', 0))
        households = float(request.form.get('households', 0))
        income = float(request.form.get('income', 0))
        proximity = int(request.form.get('proximity', -1))  # Giá trị proximity

        # Kiểm tra giá trị
        if proximity < 0 or proximity > 4:
            return render_template('index.html', result="Invalid proximity selected")

        if longitude < -124.35 or longitude > -114.31:
            return render_template('index.html', result="Invalid longitude!This longitude is not belong to California")
        
        if latitude < 32.54 or latitude > 41.95:
            return render_template('index.html', result="Invalid latitude!This latitude is not belong to California")
        # Gọi hàm dự đoán
        predicted_price = pred(
            longitude, latitude, age, rooms, bedrooms, population, households, income, proximity
        )

        # Hiển thị kết quả
        return render_template('index.html', result=f"Predicted Price: ${predicted_price:.2f}")
    except ValueError as e:
        # Xử lý lỗi ValueError khi dữ liệu không hợp lệ
        return render_template('index.html', result="Invalid input. Please enter all fields correctly.")
    except Exception as e:
        # Xử lý lỗi khác
        return render_template('index.html', result="An error occurred. Please try again.")

# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True)