# Phụ lục: Toán học cho học sâu
<a id="chap_appendix_math"></a>

**Brent Werness** (*Amazon*), **Rachel Hu** (*Amazon*), và các tác giả của cuốn sách này


Một trong những điều tuyệt vời của học sâu hiện đại là phần lớn nội dung của nó có thể được hiểu và sử dụng mà không cần hiểu đầy đủ lớp toán học bên dưới. Đây là một dấu hiệu cho thấy lĩnh vực này đang trưởng thành. Cũng như hầu hết nhà phát triển phần mềm không còn cần bận tâm đến lý thuyết về các hàm tính được, những người thực hành học sâu cũng không nên phải bận tâm đến các nền tảng lý thuyết của học cực đại hóa hợp lý.

Nhưng chúng ta vẫn chưa hoàn toàn đạt tới điều đó.

Trong thực tế, đôi khi bạn sẽ cần hiểu các lựa chọn kiến trúc ảnh hưởng đến dòng gradient như thế nào, hoặc những giả định ngầm mà bạn đưa ra khi huấn luyện với một hàm mất mát nhất định. Bạn có thể cần biết rốt cuộc entropy đo cái gì, và nó có thể giúp bạn hiểu chính xác bits-per-character trong mô hình của mình nghĩa là gì. Tất cả những điều này đều đòi hỏi hiểu biết toán học sâu hơn.

Phụ lục này nhằm cung cấp cho bạn nền tảng toán học cần thiết để hiểu lý thuyết cốt lõi của học sâu hiện đại, nhưng nó không toàn diện. Chúng ta sẽ bắt đầu bằng cách xem xét đại số tuyến tính sâu hơn. Chúng ta phát triển một cách hiểu hình học về tất cả các đối tượng và phép toán đại số tuyến tính phổ biến, nhờ đó có thể hình dung tác động của các phép biến đổi khác nhau lên dữ liệu. Một yếu tố then chốt là xây dựng các kiến thức cơ bản về phân rã trị riêng.

Tiếp theo, chúng ta phát triển lý thuyết giải tích vi phân tới mức có thể hiểu đầy đủ tại sao gradient là hướng giảm dốc nhất, và tại sao lan truyền ngược có dạng như nó vốn có. Sau đó, giải tích tích phân được thảo luận ở mức cần thiết để hỗ trợ chủ đề tiếp theo của chúng ta, lý thuyết xác suất.

Các vấn đề gặp trong thực tế thường không chắc chắn, vì vậy chúng ta cần một ngôn ngữ để nói về những điều bất định. Chúng ta ôn lại lý thuyết về biến ngẫu nhiên và các phân phối thường gặp nhất để có thể thảo luận về mô hình theo nghĩa xác suất. Điều này cung cấp nền tảng cho bộ phân loại naive Bayes, một kỹ thuật phân loại xác suất.

Liên quan chặt chẽ đến lý thuyết xác suất là nghiên cứu về thống kê. Mặc dù thống kê là một lĩnh vực quá rộng để có thể trình bày thỏa đáng trong một phần ngắn, chúng ta sẽ giới thiệu các khái niệm nền tảng mà mọi người thực hành học máy nên biết, đặc biệt là: đánh giá và so sánh các bộ ước lượng, thực hiện kiểm định giả thuyết, và xây dựng khoảng tin cậy.

Cuối cùng, chúng ta chuyển sang chủ đề lý thuyết thông tin, là nghiên cứu toán học về lưu trữ và truyền tải thông tin. Chủ đề này cung cấp ngôn ngữ cốt lõi để chúng ta có thể thảo luận định lượng về lượng thông tin mà một mô hình nắm giữ trên một miền diễn ngôn.

Gộp lại, những nội dung này tạo thành lõi của các khái niệm toán học cần thiết để bắt đầu con đường hướng tới một hiểu biết sâu sắc về học sâu.

```toc
:maxdepth: 2

geometry-linear-algebraic-ops
eigendecomposition
single-variable-calculus
multivariable-calculus
integral-calculus
random-variables
maximum-likelihood
distributions
naive-bayes
statistics
information-theory
```
