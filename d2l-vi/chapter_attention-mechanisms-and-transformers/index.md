# Cơ Chế Attention và Transformer
<a id="chap_attention-and-transformers"></a>


Những năm đầu của sự bùng nổ deep learning chủ yếu được thúc đẩy
bởi các kết quả đạt được bằng cách sử dụng kiến trúc perceptron nhiều lớp,
mạng tích chập và mạng hồi tiếp.
Đáng chú ý là các kiến trúc mô hình làm nền tảng
cho nhiều đột phá của deep learning trong thập niên 2010
đã thay đổi đáng chú ý rất ít so với
các tiền thân của chúng mặc dù đã trải qua gần 30 năm.
Trong khi nhiều đổi mới phương pháp mới
đã tìm được đường vào trong bộ công cụ của hầu hết các nhà thực hành---các hàm kích hoạt ReLU,
lớp phần dư, chuẩn hóa batch, dropout,
và các lịch trình tốc độ học thích nghi---kiến trúc cơ bản
cốt lõi vẫn có thể nhận ra rõ ràng là
các triển khai được mở rộng của các ý tưởng cổ điển.
Mặc dù hàng nghìn bài báo đề xuất các ý tưởng thay thế,
các mô hình giống mạng nơ-ron tích chập cổ điển ([chap_cnn](#chap_cnn))
vẫn giữ trạng thái *state-of-the-art* trong thị giác máy tính
và các mô hình giống thiết kế gốc của Sepp Hochreiter
cho mạng nơ-ron hồi tiếp LSTM ([sec_lstm](#sec_lstm)),
chiếm ưu thế trong hầu hết các ứng dụng xử lý ngôn ngữ tự nhiên.
Có thể lập luận rằng, đến thời điểm đó, sự nổi lên nhanh chóng của deep learning
dường như chủ yếu là do sự thay đổi
trong các tài nguyên tính toán có sẵn
(nhờ đổi mới trong tính toán song song với GPU)
và sự sẵn có của các tài nguyên dữ liệu khổng lồ
(nhờ bộ nhớ giá rẻ và dịch vụ Internet).
Trong khi các yếu tố này thực sự vẫn có thể là động lực chính
đằng sau sức mạnh ngày càng tăng của công nghệ này
chúng ta cũng đang chứng kiến, sau rất lâu,
một sự thay đổi căn bản trong bối cảnh của các kiến trúc chiếm ưu thế.

Vào thời điểm hiện tại, các mô hình chiếm ưu thế
cho hầu hết các nhiệm vụ xử lý ngôn ngữ tự nhiên
đều dựa trên kiến trúc Transformer.
Cho bất kỳ nhiệm vụ mới nào trong xử lý ngôn ngữ tự nhiên, cách tiếp cận mặc định đầu tiên
là lấy một mô hình đã được huấn luyện trước dựa trên Transformer lớn,
(ví dụ: BERT [Devlin.Chang.Lee.ea.2018], ELECTRA [clark2019electra], RoBERTa [Liu.Ott.Goyal.ea.2019], hoặc Longformer [beltagy2020longformer])
điều chỉnh các lớp đầu ra khi cần thiết,
và tinh chỉnh mô hình trên dữ liệu có sẵn
cho nhiệm vụ hạ lưu. 
Nếu bạn đã chú ý đến vài năm gần đây
của những tin tức hào hứng tập trung vào
các mô hình ngôn ngữ lớn của OpenAI, thì bạn đã theo dõi một cuộc trò chuyện
tập trung vào các mô hình dựa trên Transformer GPT-2 và GPT-3 [Radford.Wu.Child.ea.2019, brown2020language].
Trong khi đó, vision Transformer đã nổi lên
như mô hình mặc định cho các nhiệm vụ thị giác đa dạng,
bao gồm nhận dạng ảnh, phát hiện đối tượng,
phân đoạn ngữ nghĩa và siêu phân giải [Dosovitskiy.Beyer.Kolesnikov.ea.2021, liu2021swin].
Các Transformer cũng xuất hiện như các phương pháp cạnh tranh
cho nhận dạng giọng nói [gulati2020conformer],
học tăng cường [chen2021decision],
và mạng nơ-ron đồ thị [dwivedi2020generalization].

Ý tưởng cốt lõi đằng sau mô hình Transformer là *cơ chế attention*,
một đổi mới ban đầu được hình dung như một cải tiến
cho các RNN bộ mã hóa--bộ giải mã áp dụng cho các ứng dụng chuỗi-sang-chuỗi,
như dịch máy [Bahdanau.Cho.Bengio.2014].
Bạn có thể nhớ lại rằng trong các mô hình chuỗi-sang-chuỗi đầu tiên
cho dịch máy [Sutskever.Vinyals.Le.2014],
toàn bộ đầu vào được bộ mã hóa nén
thành một vector có độ dài cố định duy nhất để đưa vào bộ giải mã.
Trực giác đằng sau attention là thay vì nén đầu vào,
sẽ tốt hơn cho bộ giải mã khi xem lại chuỗi đầu vào tại mỗi bước.
Hơn nữa, thay vì luôn thấy cùng một biểu diễn của đầu vào,
người ta có thể tưởng tượng rằng bộ giải mã nên tập trung có chọn lọc
vào các phần cụ thể của chuỗi đầu vào tại các bước giải mã cụ thể.
Cơ chế attention của Bahdanau cung cấp một phương tiện đơn giản
qua đó bộ giải mã có thể *tập trung* động vào các
phần khác nhau của đầu vào tại mỗi bước giải mã.
Ý tưởng cấp cao là bộ mã hóa có thể tạo ra một biểu diễn
có độ dài bằng với chuỗi đầu vào gốc.
Sau đó, tại thời gian giải mã, bộ giải mã có thể (thông qua một cơ chế điều khiển nào đó)
nhận làm đầu vào một vector ngữ cảnh gồm tổng có trọng số
của các biểu diễn trên đầu vào tại mỗi bước thời gian.
Theo trực giác, các trọng số xác định mức độ
"tập trung" ngữ cảnh của mỗi bước vào mỗi token đầu vào,
và điều quan trọng là làm cho quá trình
gán các trọng số này có thể khả vi
để nó có thể được học cùng với
tất cả các tham số mạng nơ-ron khác.

Ban đầu, ý tưởng này là một cải tiến đáng kể
cho các mạng nơ-ron hồi tiếp
đã chiếm ưu thế trong các ứng dụng dịch máy.
Các mô hình hoạt động tốt hơn
các kiến trúc chuỗi-sang-chuỗi bộ mã hóa--bộ giải mã gốc.
Hơn nữa, các nhà nghiên cứu nhận thấy rằng một số hiểu biết định tính tốt
đôi khi xuất hiện từ việc kiểm tra mẫu trọng số attention.
Trong các nhiệm vụ dịch thuật, các mô hình attention
thường gán trọng số attention cao cho các từ đồng nghĩa liên ngôn ngữ
khi tạo ra các từ tương ứng trong ngôn ngữ đích.
Ví dụ, khi dịch câu "my feet hurt"
sang "j'ai mal au pieds", mạng nơ-ron có thể gán
trọng số attention cao cho biểu diễn của "feet"
khi tạo ra từ tiếng Pháp tương ứng "pieds".
Những hiểu biết này đã thúc đẩy các tuyên bố rằng các mô hình attention cung cấp "khả năng diễn giải"
mặc dù chính xác thì các trọng số attention có nghĩa là gì---tức là,
làm thế nào, nếu có, chúng nên được *diễn giải* vẫn là chủ đề nghiên cứu mù mờ.

Tuy nhiên, các cơ chế attention sớm nổi lên như những mối quan tâm đáng kể hơn,
vượt ra ngoài sự hữu ích của chúng như một cải tiến cho các mạng nơ-ron hồi tiếp bộ mã hóa--bộ giải mã
và sự hữu ích giả định của chúng trong việc chọn các đầu vào nổi bật.
Vaswani.Shazeer.Parmar.ea.2017 đề xuất
kiến trúc Transformer cho dịch máy,
loại bỏ hoàn toàn các kết nối hồi tiếp,
và thay vào đó dựa vào các cơ chế attention được sắp xếp khéo léo
để nắm bắt tất cả các mối quan hệ giữa các token đầu vào và đầu ra.
Kiến trúc hoạt động đáng kể tốt,
và đến năm 2018 Transformer bắt đầu xuất hiện
trong phần lớn các hệ thống xử lý ngôn ngữ tự nhiên state-of-the-art.
Hơn nữa, đồng thời, thực hành chiếm ưu thế trong xử lý ngôn ngữ tự nhiên
trở thành việc huấn luyện trước các mô hình quy mô lớn
trên các kho ngữ liệu nền tảng chung khổng lồ
để tối ưu hóa một số mục tiêu tiền huấn luyện tự giám sát,
và sau đó tinh chỉnh các mô hình này
sử dụng dữ liệu hạ lưu có sẵn.
Khoảng cách giữa Transformer và các kiến trúc truyền thống
ngày càng lớn khi áp dụng trong mô hình tiền huấn luyện này,
và do đó sự thăng tiến của Transformer trùng hợp
với sự thăng tiến của các mô hình đã được huấn luyện trước quy mô lớn như vậy,
đôi khi được gọi là *mô hình nền tảng* [bommasani2021opportunities].


Trong chương này, chúng ta giới thiệu các mô hình attention,
bắt đầu từ những trực giác cơ bản nhất
và các phiên bản đơn giản nhất của ý tưởng.
Sau đó chúng ta tiến dần đến kiến trúc Transformer,
vision Transformer, và bối cảnh
của các mô hình đã được huấn luyện trước dựa trên Transformer hiện đại.

```toc
:maxdepth: 2

queries-keys-values
attention-pooling
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
vision-transformer
large-pretraining-transformers
```
