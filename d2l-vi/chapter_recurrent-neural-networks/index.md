# Mạng Nơ-ron Hồi quy
<a id="chap_rnn"></a>

Cho đến nay, chúng ta chủ yếu tập trung vào dữ liệu có độ dài cố định.
Khi giới thiệu hồi quy tuyến tính và logistic
trong [chap_regression](#chap_regression) và [chap_classification](#chap_classification)
và perceptron đa lớp trong [chap_perceptrons](#chap_perceptrons),
chúng ta đã vui lòng giả định rằng mỗi vector đặc trưng $\mathbf{x}_i$
bao gồm một số thành phần cố định $x_1, \dots, x_d$,
trong đó mỗi đặc trưng số $x_j$
tương ứng với một thuộc tính cụ thể.
Các tập dữ liệu này đôi khi được gọi là *bảng*,
bởi vì chúng có thể được sắp xếp trong các bảng,
trong đó mỗi mẫu $i$ có hàng riêng của nó,
và mỗi thuộc tính có cột riêng của nó.
Điều quan trọng là, với dữ liệu bảng, chúng ta hiếm khi
giả định bất kỳ cấu trúc cụ thể nào trên các cột.

Sau đó, trong [chap_cnn](#chap_cnn),
chúng ta chuyển sang dữ liệu ảnh, nơi đầu vào bao gồm
các giá trị pixel thô tại mỗi tọa độ trong ảnh.
Dữ liệu ảnh hầu như không phù hợp với tiêu chuẩn
của một tập dữ liệu bảng điển hình.
Ở đó, chúng ta cần kêu gọi mạng nơ-ron tích chập (CNN)
để xử lý cấu trúc phân cấp và các bất biến.
Tuy nhiên, dữ liệu của chúng ta vẫn có độ dài cố định.
Mỗi ảnh Fashion-MNIST được biểu diễn
dưới dạng lưới $28 \times 28$ giá trị pixel.
Hơn nữa, mục tiêu của chúng ta là phát triển một mô hình
nhìn vào chỉ một ảnh và sau đó
xuất ra một dự đoán duy nhất.
Nhưng chúng ta nên làm gì khi đối mặt với
một chuỗi ảnh, như trong video,
hoặc khi được giao nhiệm vụ tạo ra
một dự đoán có cấu trúc tuần tự,
như trong trường hợp chú thích ảnh?

Rất nhiều nhiệm vụ học tập đòi hỏi phải xử lý dữ liệu tuần tự.
Chú thích ảnh, tổng hợp giọng nói và tạo nhạc
tất cả đều yêu cầu mô hình tạo ra các đầu ra bao gồm các chuỗi.
Trong các lĩnh vực khác, chẳng hạn như dự đoán chuỗi thời gian,
phân tích video và truy xuất thông tin âm nhạc,
một mô hình phải học từ các đầu vào là chuỗi.
Những yêu cầu này thường phát sinh đồng thời:
các nhiệm vụ như dịch các đoạn văn bản
từ ngôn ngữ tự nhiên này sang ngôn ngữ khác,
tham gia vào hội thoại, hoặc điều khiển robot,
đòi hỏi các mô hình vừa nhận vừa xuất ra
dữ liệu có cấu trúc tuần tự.


Mạng nơ-ron hồi quy (RNN) là các mô hình deep learning
nắm bắt động lực học của các chuỗi qua
các kết nối *hồi quy*, có thể được coi là
các chu kỳ trong mạng nút.
Điều này có vẻ phản trực giác lúc đầu.
Dù sao, chính tính chất feedforward của mạng nơ-ron
làm cho thứ tự tính toán không mơ hồ.
Tuy nhiên, các cạnh hồi quy được định nghĩa theo cách chính xác
đảm bảo rằng không có sự mơ hồ nào có thể phát sinh.
Mạng nơ-ron hồi quy được *khai triển* qua các bước thời gian (hoặc bước chuỗi),
với các tham số cơ bản *giống nhau* được áp dụng tại mỗi bước.
Trong khi các kết nối tiêu chuẩn được áp dụng *đồng bộ*
để lan truyền các kích hoạt của mỗi lớp
đến lớp tiếp theo *tại cùng bước thời gian*,
các kết nối hồi quy là *động*,
truyền thông tin qua các bước thời gian liền kề.
Như quan điểm khai triển trong [fig_unfolded-rnn](#fig_unfolded-rnn) tiết lộ,
RNN có thể được coi là mạng nơ-ron feedforward
trong đó các tham số của mỗi lớp (cả thông thường và hồi quy)
được chia sẻ qua các bước thời gian.


![Bên trái, các kết nối hồi quy được mô tả qua các cạnh tuần hoàn. Bên phải, chúng ta khai triển RNN qua các bước thời gian. Ở đây, các cạnh hồi quy trải qua các bước thời gian liền kề, trong khi các kết nối thông thường được tính toán đồng bộ.](../img/unfolded-rnn.svg)
<a id="fig_unfolded-rnn"></a>


Giống như mạng nơ-ron rộng hơn,
RNN có một lịch sử lâu dài xuyên qua các ngành,
bắt nguồn là các mô hình não bộ được phổ biến
bởi các nhà khoa học nhận thức và sau đó được áp dụng
như các công cụ mô hình hóa thực tế được sử dụng
bởi cộng đồng machine learning.
Như chúng ta làm cho deep learning rộng hơn,
trong cuốn sách này chúng ta áp dụng quan điểm machine learning,
tập trung vào RNN như các công cụ thực tế đã nổi lên
phổ biến trong những năm 2010 nhờ
các kết quả đột phá về các nhiệm vụ đa dạng
như nhận dạng chữ viết tay [graves2008novel],
dịch máy [Sutskever.Vinyals.Le.2014],
và nhận dạng chẩn đoán y tế [Lipton.Kale.2016].
Chúng tôi chỉ độc giả quan tâm đến thêm
tài liệu nền tảng đến một
đánh giá toàn diện có sẵn công khai [Lipton.Berkowitz.Elkan.2015].
Chúng tôi cũng lưu ý rằng tính tuần tự không phải là độc quyền của RNN.
Ví dụ, CNN mà chúng ta đã giới thiệu
có thể được điều chỉnh để xử lý dữ liệu có độ dài thay đổi,
ví dụ: ảnh có độ phân giải thay đổi.
Hơn nữa, RNN gần đây đã nhường đáng kể
thị phần cho các mô hình Transformer,
sẽ được đề cập trong [chap_attention-and-transformers](#chap_attention-and-transformers).
Tuy nhiên, RNN đã nổi lên như các mô hình mặc định
để xử lý cấu trúc tuần tự phức tạp trong deep learning,
và vẫn là các mô hình chủ đạo cho mô hình hóa tuần tự cho đến nay.
Những câu chuyện của RNN và mô hình hóa chuỗi
được liên kết chặt chẽ với nhau, và đây cũng là
một chương về những kiến thức cơ bản của các bài toán mô hình hóa chuỗi
cũng như là một chương về RNN.


Một nhận thức quan trọng đã mở đường cho một cuộc cách mạng trong mô hình hóa chuỗi.
Trong khi đầu vào và mục tiêu cho nhiều nhiệm vụ cơ bản trong machine learning
không thể dễ dàng được biểu diễn dưới dạng các vector có độ dài cố định,
chúng thường có thể được biểu diễn dưới dạng
các chuỗi có độ dài thay đổi của các vector có độ dài cố định.
Ví dụ, tài liệu có thể được biểu diễn dưới dạng chuỗi các từ;
hồ sơ y tế thường có thể được biểu diễn dưới dạng chuỗi sự kiện
(gặp gỡ, thuốc, thủ thuật, xét nghiệm, chẩn đoán);
video có thể được biểu diễn dưới dạng chuỗi hình ảnh tĩnh có độ dài thay đổi.


Trong khi các mô hình chuỗi đã xuất hiện trong nhiều lĩnh vực ứng dụng,
nghiên cứu cơ bản trong lĩnh vực này chủ yếu được thúc đẩy
bởi những tiến bộ trong các nhiệm vụ cốt lõi trong xử lý ngôn ngữ tự nhiên.
Do đó, trong chương này, chúng ta sẽ tập trung
trình bày và ví dụ của mình vào dữ liệu văn bản.
Nếu bạn nắm được những ví dụ này,
thì việc áp dụng các mô hình cho các phương thức dữ liệu khác
sẽ tương đối đơn giản.
Trong các phần tiếp theo, chúng ta giới thiệu ký hiệu cơ bản
cho các chuỗi và một số thước đo đánh giá
để đánh giá chất lượng đầu ra mô hình có cấu trúc tuần tự.
Sau đó, chúng ta thảo luận các khái niệm cơ bản của mô hình ngôn ngữ
và sử dụng thảo luận này để thúc đẩy các mô hình RNN đầu tiên của chúng ta.
Cuối cùng, chúng ta mô tả phương pháp tính gradient
khi lan truyền ngược qua RNN và khám phá một số thách thức
thường gặp phải khi huấn luyện các mạng như vậy,
thúc đẩy các kiến trúc RNN hiện đại sẽ theo sau
trong [chap_modern_rnn](#chap_modern_rnn).

```toc
:maxdepth: 2

sequence
text-sequence
language-model
rnn
rnn-scratch
rnn-concise
bptt
```
