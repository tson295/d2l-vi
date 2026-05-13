# Mạng Nơ-ron Hồi quy Hiện đại
<a id="chap_modern_rnn"></a>

Chương trước đã giới thiệu các ý tưởng then chốt
đằng sau mạng nơ-ron hồi quy (RNN).
Tuy nhiên, cũng giống như với mạng nơ-ron tích chập,
đã có rất nhiều đổi mới
trong các kiến trúc RNN, dẫn đến một số thiết kế
phức tạp đã chứng tỏ thành công trong thực tế.
Cụ thể, các thiết kế phổ biến nhất
có các cơ chế để giảm nhẹ sự bất ổn số học khét tiếng
mà RNN phải đối mặt,
điển hình là gradient tiêu biến và bùng nổ.
Nhắc lại rằng trong [chap_rnn](#chap_rnn), chúng ta đã xử lý
gradient bùng nổ bằng cách áp dụng một heuristic
cắt gradient khá thô.
Bất chấp hiệu quả của mẹo này,
nó vẫn để ngỏ vấn đề gradient tiêu biến.

Trong chương này, chúng ta giới thiệu các ý tưởng then chốt đằng sau
những kiến trúc RNN thành công nhất cho dữ liệu chuỗi,
bắt nguồn từ hai bài báo.
Bài báo thứ nhất, *Long Short-Term Memory* [Hochreiter.Schmidhuber.1997],
giới thiệu *ô nhớ*, một đơn vị tính toán thay thế
các nút truyền thống trong lớp ẩn của mạng.
Với các ô nhớ này, mạng có thể
vượt qua những khó khăn trong huấn luyện
mà các mạng hồi quy trước đó gặp phải.
Về trực giác, ô nhớ tránh
vấn đề gradient tiêu biến
bằng cách giữ các giá trị trong trạng thái nội bộ của từng ô nhớ
lan truyền dọc theo một cạnh hồi quy có trọng số 1
qua nhiều bước thời gian liên tiếp.
Một tập các cổng nhân giúp mạng
xác định không chỉ những đầu vào nào được phép đi vào
trạng thái nhớ,
mà còn khi nào nội dung của trạng thái nhớ
nên ảnh hưởng đến đầu ra của mô hình.

Bài báo thứ hai, *Bidirectional Recurrent Neural Networks* [Schuster.Paliwal.1997],
giới thiệu một kiến trúc trong đó thông tin
từ cả tương lai (các bước thời gian tiếp sau)
và quá khứ (các bước thời gian trước đó)
được dùng để xác định đầu ra
tại bất kỳ điểm nào trong chuỗi.
Điều này trái ngược với các mạng trước đây,
trong đó chỉ đầu vào trong quá khứ mới có thể ảnh hưởng đến đầu ra.
RNN hai chiều đã trở thành một thành phần chủ lực
cho các tác vụ gán nhãn chuỗi trong xử lý ngôn ngữ tự nhiên,
cùng vô số tác vụ khác.
May mắn thay, hai đổi mới này không loại trừ lẫn nhau,
và đã được kết hợp thành công cho phân loại âm vị
[Graves.Schmidhuber.2005] và nhận dạng chữ viết tay [graves2008novel].


Các phần đầu tiên trong chương này sẽ giải thích kiến trúc LSTM,
một phiên bản nhẹ hơn gọi là gated recurrent unit (GRU),
các ý tưởng then chốt đằng sau RNN hai chiều
và một giải thích ngắn gọn về cách các lớp RNN
được xếp chồng với nhau để tạo thành RNN sâu.
Sau đó, chúng ta sẽ khám phá ứng dụng của RNN
trong các tác vụ chuỗi-sang-chuỗi,
giới thiệu dịch máy
cùng các ý tưởng then chốt như kiến trúc *encoder--decoder* và *beam search*.

```toc
:maxdepth: 2

lstm
gru
deep-rnn
bi-rnn
machine-translation-and-dataset
encoder-decoder
seq2seq
beam-search
```
