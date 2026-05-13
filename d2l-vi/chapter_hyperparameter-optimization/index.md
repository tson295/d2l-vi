# Tối ưu hóa siêu tham số
<a id="chap_hyperopt"></a>

**Aaron Klein** (*Amazon*), **Matthias Seeger** (*Amazon*), và **Cedric Archambeau** (*Amazon*)

Hiệu năng của mọi mô hình học máy phụ thuộc vào các siêu tham số của nó.
Chúng kiểm soát thuật toán học hoặc cấu trúc của mô hình thống kê
bên dưới. Tuy nhiên, trong thực tế không có cách tổng quát nào để chọn siêu tham số.
Thay vào đó, các siêu tham số thường được đặt theo kiểu thử-sai
hoặc đôi khi được người thực hành để nguyên giá trị mặc định, dẫn đến
khả năng khái quát hóa dưới mức tối ưu.

Tối ưu hóa siêu tham số cung cấp một cách tiếp cận có hệ thống cho bài toán này, bằng cách
đưa nó về một bài toán tối ưu hóa: một tập siêu tham số tốt nên (ít nhất)
tối thiểu hóa lỗi validation. So với hầu hết các bài toán tối ưu hóa khác
phát sinh trong học máy, tối ưu hóa siêu tham số là một bài toán lồng nhau, trong đó
mỗi lần lặp yêu cầu huấn luyện và xác thực một mô hình học máy.

Trong chương này, trước tiên chúng ta sẽ giới thiệu các kiến thức cơ bản về
tối ưu hóa siêu tham số. Chúng ta cũng sẽ trình bày một số tiến bộ gần đây giúp cải thiện
hiệu quả tổng thể của tối ưu hóa siêu tham số bằng cách khai thác các
proxy rẻ để đánh giá của hàm mục tiêu gốc. Đến cuối chương này, bạn
nên có thể áp dụng các kỹ thuật tối ưu hóa siêu tham số tiên tiến
để tối ưu hóa siêu tham số của thuật toán học máy của riêng mình.

```toc
:maxdepth: 2

hyperopt-intro
hyperopt-api
rs-async.md
sh-intro
sh-async
```
