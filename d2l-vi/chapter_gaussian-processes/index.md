# Gaussian Process
<a id="chap_gp"></a>

**Andrew Gordon Wilson** (*New York University and Amazon*)


Gaussian process (GP) xuất hiện ở khắp nơi. Bạn đã gặp nhiều ví dụ về GP mà không nhận ra. Bất kỳ mô hình nào tuyến tính theo các tham số của nó với một phân phối Gaussian trên các tham số đều là một Gaussian process. Lớp này bao trùm các mô hình rời rạc, bao gồm random walk và quá trình tự hồi quy, cũng như các mô hình liên tục, bao gồm mô hình hồi quy tuyến tính Bayes, đa thức, chuỗi Fourier, hàm cơ sở xuyên tâm, và thậm chí cả mạng nơ-ron với số lượng đơn vị ẩn vô hạn. Có một câu đùa quen thuộc rằng "mọi thứ đều là một trường hợp đặc biệt của Gaussian process".

Việc học về Gaussian process quan trọng vì ba lý do: (1) chúng cung cấp một góc nhìn _không gian hàm_ về mô hình hóa, giúp việc hiểu nhiều lớp mô hình, bao gồm mạng nơ-ron sâu, trở nên dễ tiếp cận hơn nhiều; (2) chúng có phạm vi ứng dụng đặc biệt rộng, nơi chúng đạt trạng thái tiên tiến nhất, bao gồm active learning, học siêu tham số, auto-ML và hồi quy không-thời gian; (3) trong vài năm gần đây, các tiến bộ thuật toán đã làm cho Gaussian process ngày càng có khả năng mở rộng và phù hợp hơn, hài hòa với học sâu thông qua các framework như [GPyTorch](https://gpytorch.ai) [Gardner.Pleiss.Weinberger.Bindel.Wilson.2018]. Thật vậy, GP và mạng nơ-ron sâu không phải là các hướng tiếp cận cạnh tranh, mà bổ sung mạnh mẽ cho nhau và có thể được kết hợp rất hiệu quả. Các tiến bộ thuật toán này không chỉ liên quan đến Gaussian process, mà còn cung cấp một nền tảng về phương pháp số hữu ích rộng rãi trong học sâu.

Trong chương này, chúng ta giới thiệu Gaussian process. Trong notebook nhập môn, chúng ta bắt đầu bằng cách suy luận trực giác về Gaussian process là gì và cách chúng trực tiếp mô hình hóa hàm. Trong notebook về prior, chúng ta tập trung vào cách chỉ định các prior Gaussian process. Chúng ta kết nối trực tiếp cách tiếp cận không gian trọng số truyền thống trong mô hình hóa với không gian hàm, điều này sẽ giúp chúng ta lập luận về việc xây dựng và hiểu các mô hình học máy, bao gồm mạng nơ-ron sâu. Sau đó, chúng ta giới thiệu các hàm hiệp phương sai phổ biến, còn gọi là _kernel_, vốn kiểm soát các tính chất khái quát hóa của một Gaussian process. Một GP với một kernel cho trước định nghĩa một prior trên các hàm. Trong notebook suy luận, chúng ta sẽ trình bày cách dùng dữ liệu để suy ra một _posterior_, nhằm đưa ra dự đoán. Notebook này chứa code từ đầu để dự đoán bằng một Gaussian process, cũng như phần giới thiệu về GPyTorch. Trong các notebook sắp tới, chúng ta sẽ giới thiệu phần tính toán số đằng sau Gaussian process, hữu ích cho việc mở rộng Gaussian process nhưng cũng là một nền tảng tổng quát mạnh cho học sâu, và các trường hợp sử dụng nâng cao như tinh chỉnh siêu tham số trong học sâu. Các ví dụ của chúng ta sẽ dùng GPyTorch, công cụ giúp Gaussian process có khả năng mở rộng và được tích hợp chặt chẽ với chức năng học sâu và PyTorch.

```toc
:maxdepth: 2

gp-intro
gp-priors
gp-inference
```
