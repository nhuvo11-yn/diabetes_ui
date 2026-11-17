# HƯỚNG DẪN CHO NHÓM (tiếng Việt)

Repo này đã **chạy được hoàn chỉnh**: train xong, app 4 trang, tài liệu nộp bài. Việc của nhóm là: chạy lại, hiểu, chỉnh sửa, deploy, quay video, điền các chỗ `<...>`.

## 1. Flow tổng thể

```         
data/*.csv ──► train.py ──► models/pipeline.joblib + model_meta.json
                  │               │
                  │               └──► app.py + pages/* (Streamlit)  ──► Deploy ──► link công khai
                  └──► reports/*.csv (so sánh model, fairness, importance, điểm dự đoán test)
```

1.  `train.py` đọc CSV → bỏ dòng trùng → chia 60/20/20 → train 4 model (Logistic Regression và Gradient Boosting, mỗi loại với 21 biến và 7 biến) → chọn threshold trên tập validation → đánh giá trên test → tính fairness có khoảng tin cậy → lưu mọi thứ.
2.  App chỉ **đọc** kết quả đã lưu, không train lại.
3.  Mô hình trong app: Gradient Boosting với **7 câu hỏi** (tuổi, chiều cao/cân nặng → BMI, cao huyết áp, cholesterol cao, tự đánh giá sức khỏe, khó đi lại, bệnh tim).

## 2. Chạy trên máy (5 phút)

``` bash
cd diabetes-screening-project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py                    # ~15 giây
streamlit run app.py               # mở http://localhost:8501
```

Nếu `pip install` báo không có đúng phiên bản (do Python của bạn cũ hơn): đổi `requirements.txt` sang phiên bản bạn cài được, rồi **chạy lại `python train.py`** để file `.joblib` khớp phiên bản thư viện.

## 3. Deploy (Streamlit Community Cloud)

1.  Tạo repo GitHub **public**, đẩy toàn bộ thư mục này lên (nhớ commit cả `models/` và `reports/`, vì app cần chúng).

    ``` bash
    git init && git add . && git commit -m "Diabetes screening app"
    git branch -M main && git remote add origin <url-repo> && git push -u origin main
    ```

2.  Vào share.streamlit.io → New app → chọn repo, branch `main`, file chính `app.py`.

3.  Trong *Advanced settings* chọn phiên bản Python gần với máy bạn (3.12 hoặc 3.13). Nếu app báo lỗi load model: chạy lại `python train.py` bằng đúng phiên bản thư viện trong `requirements.txt`, commit lại `models/`, push.

4.  Dán link vào README (mục Live app) và nộp.

## 4. Chia việc cho 3 người (gấp, làm song song)

| Người | Việc chính | File |
|------------------------|------------------------|------------------------|
| **A – Data & Model** | Chạy `train.py`, đọc `reports/results_summary.md`, chạy notebook EDA, hiểu pipeline và threshold | `train.py`, `core.py`, `notebooks/` |
| **B – App & Deploy** | Chạy app, chỉnh giao diện/chữ, push GitHub, deploy, test link bằng `data/sample_input.csv` | `app.py`, `pages/`, `utils.py` |
| **C – Equity & Tài liệu** | Đọc trang Health Equity, viết phần phân tích, điền README, Decision log, AI appendix, quay video | `*.md` |

Cả 3 người **đều phải đọc và giải thích được** toàn bộ pipeline, threshold, bảng fairness và cách giải thích kết quả (có thể bị hỏi khi bảo vệ).

## 5. Thứ tự làm khi thời gian gấp

1.  (30 phút) Chạy `train.py` + `streamlit run app.py`, bấm thử mọi trang.
2.  (1 giờ) Đẩy GitHub + deploy ngay (làm sớm để biết lỗi môi trường).
3.  (1–2 giờ) Đọc code, điền README (tên, link), AI appendix (phần `<...>`), thêm 4–6 câu vào notebook mục 6.
4.  (1 giờ) Quay video demo 3 phút.
5.  Cuối: đối chiếu checklist mục 7 với **đề bài gốc** (tôi không có đề nên không kiểm chứng được số trang, tên file bắt buộc, tiêu chí chấm).

## 6. Kịch bản video 3 phút (một người dùng, một quyết định)

| Thời gian | Nội dung |
|------------------------------------|------------------------------------|
| 0:00–0:25 | Vấn đề: tiểu đường chẩn đoán muộn; người dùng ở nhà cần quyết định có nên hỏi bác sĩ về xét nghiệm đường huyết. Nói câu user-decision. |
| 0:25–1:15 | Nhập hồ sơ cụ thể (ví dụ dòng 4 trong `sample_input.csv`), bấm **Analyze**, đọc kết quả và hành động khuyến nghị. |
| 1:15–1:40 | "What pushed your score": giải thích 2–3 yếu tố, nhấn mạnh không phải quan hệ nhân quả. |
| 1:40–2:25 | Trang Threshold & Cost: kéo tỷ lệ chi phí FN:FP, cho thấy recall/precision và số ca bỏ sót/báo nhầm trên 1.000 người; bấm **Use on Screening page**, quay lại chạy lại cùng hồ sơ. |
| 2:25–2:50 | Health Equity: nhóm thu nhập cao/học vấn cao bị **bỏ sót nhiều hơn**, nhóm thu nhập thấp bị **báo nhầm nhiều hơn**; đây là dấu hiệu cần xem xét, không phải kết luận phân biệt đối xử. |
| 2:50–3:00 | Giới hạn: dữ liệu Mỹ 2015, tự khai báo, không phải chẩn đoán. |

## 7. Checklist nộp bài

-   [ ] Link app công khai chạy được (thử trên điện thoại)
-   [ ] GitHub repo public (có `models/`, `reports/`, `data/`)
-   [ ] `python train.py` chạy lại được từ dữ liệu gốc
-   [ ] `models/pipeline.joblib` và `models/model_meta.json`
-   [ ] Video demo 3 phút
-   [ ] `DECISION_LOG.md` (đọc lại, sửa nếu nhóm quyết định khác)
-   [ ] `AI_USE_APPENDIX.md` (điền phần `<...>` bằng việc nhóm **thực sự** làm; không ghi sai)
-   [ ] `README.md` (điền tên, link; 1–2 trang)

## 8. Câu hỏi có thể bị hỏi (nên chuẩn bị)

-   **Tại sao không dùng accuracy?** Chỉ \~15% là dương tính; đoán "không ai bị" đã đúng \~85%.
-   **Tại sao threshold ≈ 0.185, không phải 0.5?** Chọn để tối thiểu chi phí giả định (bỏ sót nặng gấp 5 lần xét nghiệm thừa) trên tập validation; người dùng đổi được ở trang Threshold.
-   **Chi phí 5:1 lấy ở đâu?** Là giả định để mô phỏng quyết định, không phải chi phí y tế đo được. Nói rõ như vậy.
-   **Tại sao bỏ dòng trùng?** Tránh cùng một hồ sơ nằm ở cả train và test; nhưng các dòng trùng cũng có thể là những người khác nhau. Có kiểm tra độ nhạy (`--keep-duplicates`) trong Decision log.
-   **Tại sao 7 câu hỏi?** Hiệu suất gần bằng 21 biến (PR-AUC validation 0.432 so với 0.440), và người dùng tự trả lời được ở nhà.
-   **Mô hình có công bằng không?** Không đồng đều: nhóm thu nhập/học vấn cao bị bỏ sót nhiều hơn, nhóm thu nhập thấp bị báo nhầm nhiều hơn dù mô hình không dùng thu nhập. Một phần do tỷ lệ bệnh nền khác nhau giữa các nhóm.
-   **Áp dụng cho Việt Nam được không?** Không trực tiếp: dữ liệu Mỹ 2015, tự khai báo; người châu Á có thể nguy cơ ở BMI thấp hơn.
-   **SHAP đâu?** Dùng phương pháp thay từng câu trả lời bằng giá trị điển hình + permutation importance + odds ratio của logistic regression (đỡ phụ thuộc thư viện). Nếu đề bắt buộc SHAP, thêm `shap` và tính cho mô hình cây (ghi vào Decision log).

## 9. Nếu đề bài yêu cầu khác

-   Đề bắt buộc **trang riêng cho giải thích**: tách phần "What pushed your score" trong `app.py` thành `pages/`, dùng `st.session_state["result"]`.
-   Đề muốn **mô hình tuned/so sánh nhiều hơn**: thêm vào vòng lặp `for algo in ...` trong `train.py` (hàm `build_pipeline`).
-   Đổi tỷ lệ chi phí mặc định: `python train.py --cost-fn 8`.
