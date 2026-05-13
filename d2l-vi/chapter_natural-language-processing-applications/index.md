# Xử lý ngôn ngữ tự nhiên: Ứng dụng
<a id="chap_nlp_app"></a>

Chúng ta đã thấy cách biểu diễn các token trong chuỗi văn bản và huấn luyện các biểu diễn của chúng trong [chap_nlp_pretrain](#chap_nlp_pretrain).
Các biểu diễn văn bản đã tiền huấn luyện như vậy có thể được đưa vào nhiều mô hình khác nhau cho các tác vụ xử lý ngôn ngữ tự nhiên downstream khác nhau.

Thực ra,
các chương trước đã thảo luận một số ứng dụng xử lý ngôn ngữ tự nhiên
*không dùng tiền huấn luyện*,
chỉ nhằm giải thích các kiến trúc học sâu.
Chẳng hạn, trong [chap_rnn](#chap_rnn),
chúng ta đã dựa vào RNN để thiết kế các mô hình ngôn ngữ nhằm sinh văn bản giống tiểu thuyết ngắn.
Trong [chap_modern_rnn](#chap_modern_rnn) và [chap_attention-and-transformers](#chap_attention-and-transformers),
chúng ta cũng đã thiết kế các mô hình dựa trên RNN và cơ chế chú ý cho dịch máy.

Tuy nhiên, cuốn sách này không có ý định bao quát toàn diện tất cả các ứng dụng như vậy.
Thay vào đó,
trọng tâm của chúng ta là *cách áp dụng học biểu diễn (sâu) cho ngôn ngữ để giải quyết các bài toán xử lý ngôn ngữ tự nhiên*.
Với các biểu diễn văn bản đã tiền huấn luyện,
chương này sẽ khảo sát hai tác vụ xử lý ngôn ngữ tự nhiên downstream
phổ biến và có tính đại diện:
phân tích cảm xúc và suy luận ngôn ngữ tự nhiên,
lần lượt phân tích một văn bản đơn lẻ và quan hệ giữa các cặp văn bản.

![Các biểu diễn văn bản đã tiền huấn luyện có thể được đưa vào nhiều kiến trúc học sâu khác nhau cho các ứng dụng xử lý ngôn ngữ tự nhiên downstream khác nhau. Chương này tập trung vào cách thiết kế mô hình cho các ứng dụng xử lý ngôn ngữ tự nhiên downstream khác nhau.](../img/nlp-map-app.svg)
<a id="fig_nlp-map-app"></a>

Như được minh họa trong [fig_nlp-map-app](#fig_nlp-map-app),
chương này tập trung mô tả các ý tưởng cơ bản trong việc thiết kế mô hình xử lý ngôn ngữ tự nhiên bằng những loại kiến trúc học sâu khác nhau, chẳng hạn như MLP, CNN, RNN và attention.
Dù có thể kết hợp bất kỳ biểu diễn văn bản đã tiền huấn luyện nào với bất kỳ kiến trúc nào cho một trong hai ứng dụng trong [fig_nlp-map-app](#fig_nlp-map-app),
chúng ta chọn một vài tổ hợp có tính đại diện.
Cụ thể, chúng ta sẽ khảo sát các kiến trúc phổ biến dựa trên RNN và CNN cho phân tích cảm xúc.
Với suy luận ngôn ngữ tự nhiên, chúng ta chọn attention và MLP để minh họa cách phân tích các cặp văn bản.
Cuối cùng, chúng ta giới thiệu cách fine-tune một mô hình BERT đã tiền huấn luyện
cho một phạm vi rộng các ứng dụng xử lý ngôn ngữ tự nhiên,
chẳng hạn ở cấp chuỗi (phân loại văn bản đơn và phân loại cặp văn bản)
và cấp token (gán nhãn văn bản và hỏi đáp).
Như một trường hợp thực nghiệm cụ thể,
chúng ta sẽ fine-tune BERT cho suy luận ngôn ngữ tự nhiên.

Như đã giới thiệu trong [sec_bert](#sec_bert),
BERT yêu cầu rất ít thay đổi kiến trúc
cho một phạm vi rộng các ứng dụng xử lý ngôn ngữ tự nhiên.
Tuy nhiên, lợi ích này đi kèm chi phí fine-tune
một số lượng rất lớn tham số BERT cho các ứng dụng downstream.
Khi không gian hoặc thời gian bị giới hạn,
các mô hình được thiết kế dựa trên MLP, CNN, RNN và attention
khả thi hơn.
Sau đây, chúng ta bắt đầu với ứng dụng phân tích cảm xúc
và lần lượt minh họa thiết kế mô hình dựa trên RNN và CNN.

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```
