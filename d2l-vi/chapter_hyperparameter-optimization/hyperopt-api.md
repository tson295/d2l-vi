# API tối ưu hóa siêu tham số
<a id="sec_api_hpo"></a>

Trước khi đi sâu vào phương pháp luận, trước tiên chúng ta sẽ thảo luận một cấu trúc code
cơ bản cho phép triển khai hiệu quả nhiều thuật toán HPO khác nhau. Nhìn
chung, mọi thuật toán HPO được xét ở đây cần triển khai hai primitive ra
quyết định, *searching* và *scheduling*. Thứ nhất, chúng cần lấy mẫu các
cấu hình siêu tham số mới, thường liên quan đến một dạng tìm kiếm nào đó trên
không gian cấu hình. Thứ hai, với mỗi cấu hình, một thuật toán HPO cần
lập lịch đánh giá nó và quyết định cấp bao nhiêu tài nguyên cho nó. Khi
bắt đầu đánh giá một cấu hình, chúng ta sẽ gọi nó là một *trial*. Chúng ta ánh xạ
các quyết định này vào hai lớp, `HPOSearcher` và `HPOScheduler`. Trên đó,
chúng ta cũng cung cấp một lớp `HPOTuner` thực thi quá trình tối ưu hóa.

Khái niệm scheduler và searcher này cũng được triển khai trong các thư viện HPO
phổ biến, chẳng hạn Syne Tune [salinas-automl22], Ray Tune
[liaw-arxiv18] hoặc Optuna [akiba-sigkdd19].

```python
import time
from d2l import torch as d2l
from scipy import stats
```

## Searcher

Dưới đây chúng ta định nghĩa một lớp cơ sở cho searcher, cung cấp một cấu hình
ứng viên mới thông qua hàm `sample_configuration`. Một cách đơn giản để
triển khai hàm này là lấy mẫu cấu hình đều ngẫu nhiên,
như đã làm cho random search trong [sec_what_is_hpo](#sec_what_is_hpo). Các thuật toán
tinh vi hơn, chẳng hạn Bayesian optimization, sẽ đưa ra các
quyết định này dựa trên hiệu năng của các trial trước đó. Nhờ vậy, các
thuật toán này có thể lấy mẫu các ứng viên hứa hẹn hơn theo thời gian. Chúng ta thêm
hàm `update` để cập nhật lịch sử các trial trước đó, vốn sau đó có thể
được khai thác để cải thiện phân phối lấy mẫu.

```python
class HPOSearcher(d2l.HyperParameters):  
    def sample_configuration() -> dict:
        raise NotImplementedError

    def update(self, config: dict, error: float, additional_info=None):
        pass
```

Đoạn code sau cho thấy cách triển khai optimizer random search từ
phần trước trong API này. Như một mở rộng nhỏ, chúng ta cho phép người dùng
chỉ định cấu hình đầu tiên cần đánh giá thông qua `initial_config`, trong khi
các cấu hình sau được rút ngẫu nhiên.

```python
class RandomSearcher(HPOSearcher):  
    def __init__(self, config_space: dict, initial_config=None):
        self.save_hyperparameters()

    def sample_configuration(self) -> dict:
        if self.initial_config is not None:
            result = self.initial_config
            self.initial_config = None
        else:
            result = {
                name: domain.rvs()
                for name, domain in self.config_space.items()
            }
        return result
```

## Scheduler

Bên cạnh việc lấy mẫu cấu hình cho các trial mới, chúng ta cũng cần quyết định khi nào và
chạy một trial trong bao lâu. Trong thực tế, mọi quyết định này được thực hiện bởi
`HPOScheduler`, lớp ủy quyền việc chọn cấu hình mới cho một
`HPOSearcher`. Phương thức `suggest` được gọi bất cứ khi nào có tài nguyên huấn luyện
sẵn sàng. Ngoài việc gọi `sample_configuration` của searcher, nó
cũng có thể quyết định các tham số như `max_epochs` (tức huấn luyện mô hình
trong bao lâu). Phương thức `update` được gọi bất cứ khi nào một trial trả về một
quan sát mới.

```python
class HPOScheduler(d2l.HyperParameters):  
    def suggest(self) -> dict:
        raise NotImplementedError
    
    def update(self, config: dict, error: float, info=None):
        raise NotImplementedError
```

Để triển khai random search, cũng như các thuật toán HPO khác, chúng ta chỉ cần một
scheduler cơ bản lập lịch một cấu hình mới mỗi khi tài nguyên mới
sẵn sàng.

```python
class BasicScheduler(HPOScheduler):  
    def __init__(self, searcher: HPOSearcher):
        self.save_hyperparameters()

    def suggest(self) -> dict:
        return self.searcher.sample_configuration()

    def update(self, config: dict, error: float, info=None):
        self.searcher.update(config, error, additional_info=info)
```

## Tuner

Cuối cùng, chúng ta cần một thành phần chạy scheduler/searcher và thực hiện một số
ghi sổ kết quả. Đoạn code sau triển khai việc thực thi tuần tự
các trial HPO, đánh giá từng job huấn luyện một và
sẽ đóng vai trò ví dụ cơ bản. Về sau, chúng ta sẽ dùng *Syne Tune* cho các
trường hợp HPO phân tán có khả năng mở rộng hơn.

```python
class HPOTuner(d2l.HyperParameters):  
    def __init__(self, scheduler: HPOScheduler, objective: callable):
        self.save_hyperparameters()
        # Bookeeping results for plotting
        self.incumbent = None
        self.incumbent_error = None
        self.incumbent_trajectory = []
        self.cumulative_runtime = []
        self.current_runtime = 0
        self.records = []

    def run(self, number_of_trials):
        for i in range(number_of_trials):
            start_time = time.time()
            config = self.scheduler.suggest()
            print(f"Trial {i}: config = {config}")
            error = self.objective(**config)
            error = float(d2l.numpy(error.cpu()))
            self.scheduler.update(config, error)
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)
            print(f"    error = {error}, runtime = {runtime}")
```

## Ghi sổ hiệu năng của các thuật toán HPO

Với bất kỳ thuật toán HPO nào, chúng ta chủ yếu quan tâm đến cấu hình hoạt động tốt nhất
(gọi là *incumbent*) và lỗi validation của nó sau một thời lượng wall-clock
cho trước. Đây là lý do chúng ta theo dõi `runtime` mỗi lần lặp, bao gồm
cả thời gian chạy một đánh giá (lời gọi `objective`) và thời gian
ra quyết định (lời gọi `scheduler.suggest`). Sau đây, chúng ta sẽ vẽ
`cumulative_runtime` theo `incumbent_trajectory` để trực quan hóa
*any-time performance* của thuật toán HPO được định nghĩa theo `scheduler`
(và `searcher`). Điều này cho phép chúng ta định lượng không chỉ cấu hình
mà optimizer tìm được hoạt động tốt thế nào, mà còn optimizer có thể tìm nó nhanh đến đâu.

```python
@d2l.add_to_class(HPOTuner)  
def bookkeeping(self, config: dict, error: float, runtime: float):
    self.records.append({"config": config, "error": error, "runtime": runtime})
    # Check if the last hyperparameter configuration performs better 
    # than the incumbent
    if self.incumbent is None or self.incumbent_error > error:
        self.incumbent = config
        self.incumbent_error = error
    # Add current best observed performance to the optimization trajectory
    self.incumbent_trajectory.append(self.incumbent_error)
    # Update runtime
    self.current_runtime += runtime
    self.cumulative_runtime.append(self.current_runtime)
```

## Ví dụ: Tối ưu hóa siêu tham số của mạng nơ-ron tích chập

Bây giờ chúng ta dùng triển khai random search mới để tối ưu hóa
*batch size* và *learning rate* của mạng nơ-ron tích chập `LeNet`
từ [sec_lenet](#sec_lenet). Chúng ta bắt đầu bằng cách định nghĩa hàm mục tiêu, một lần nữa
sẽ là lỗi validation.

```python
def hpo_objective_lenet(learning_rate, batch_size, max_epochs=10):  
    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=max_epochs, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model=model, data=data)
    validation_error = trainer.validation_error()
    return validation_error
```

Chúng ta cũng cần định nghĩa không gian cấu hình. Hơn nữa, cấu hình đầu tiên
được đánh giá là thiết lập mặc định dùng trong [sec_lenet](#sec_lenet).

```python
config_space = {
    "learning_rate": stats.loguniform(1e-2, 1),
    "batch_size": stats.randint(32, 256),
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

Bây giờ chúng ta có thể bắt đầu random search:

```python
searcher = RandomSearcher(config_space, initial_config=initial_config)
scheduler = BasicScheduler(searcher=searcher)
tuner = HPOTuner(scheduler=scheduler, objective=hpo_objective_lenet)
tuner.run(number_of_trials=5)
```

Bên dưới, chúng ta vẽ quỹ đạo tối ưu hóa của incumbent để thu được any-time
performance của random search:

```python
board = d2l.ProgressBoard(xlabel="time", ylabel="error")
for time_stamp, error in zip(
    tuner.cumulative_runtime, tuner.incumbent_trajectory
):
    board.draw(time_stamp, error, "random search", every_n=1)
```

## So sánh các thuật toán HPO

Cũng như với thuật toán huấn luyện hoặc kiến trúc mô hình, điều quan trọng là
hiểu cách so sánh tốt nhất các thuật toán HPO khác nhau. Mỗi lần chạy HPO phụ thuộc
vào hai nguồn ngẫu nhiên chính: các hiệu ứng ngẫu nhiên của quá trình huấn luyện,
chẳng hạn khởi tạo trọng số ngẫu nhiên hoặc thứ tự mini-batch, và tính ngẫu nhiên nội tại
của chính thuật toán HPO, chẳng hạn lấy mẫu ngẫu nhiên trong random
search. Do đó, khi so sánh các thuật toán khác nhau, điều then chốt là chạy mỗi
thí nghiệm nhiều lần và báo cáo các thống kê, chẳng hạn trung bình hoặc trung vị, trên
một quần thể gồm nhiều lần lặp của một thuật toán dựa trên các seed khác nhau
của bộ sinh số ngẫu nhiên.

Để minh họa điều này, chúng ta so sánh random search (xem [sec_rs](#sec_rs)) và Bayesian
optimization [snoek-nips12] khi tinh chỉnh siêu tham số của một mạng nơ-ron
feed-forward. Mỗi thuật toán được đánh giá
$50$ lần với một seed ngẫu nhiên khác nhau. Đường liền biểu thị hiệu năng trung bình
của incumbent trên $50$ lần lặp này và đường đứt biểu thị
độ lệch chuẩn. Ta có thể thấy random search và Bayesian optimization
hoạt động xấp xỉ như nhau đến khoảng ~1000 giây, nhưng Bayesian optimization có thể
tận dụng quan sát trong quá khứ để xác định các cấu hình tốt hơn và do đó
nhanh chóng vượt trội random search sau đó.


![Ví dụ đồ thị any-time performance để so sánh hai thuật toán A và B.](../img/example_anytime_performance.svg)
<a id="example_anytime_performance"></a>

## Tóm tắt

Phần này đã trình bày một giao diện đơn giản nhưng linh hoạt để triển khai nhiều thuật toán HPO
mà chúng ta sẽ xem trong chương này. Các giao diện tương tự có thể được tìm thấy
trong các framework HPO mã nguồn mở phổ biến. Chúng ta cũng đã xem cách so sánh các thuật toán HPO
và những cạm bẫy tiềm ẩn cần lưu ý.

## Bài tập

1. Mục tiêu của bài tập này là triển khai hàm mục tiêu cho một bài toán HPO khó hơn một chút, và chạy các thí nghiệm thực tế hơn. Chúng ta sẽ dùng MLP hai tầng ẩn `DropoutMLP` được triển khai trong [sec_dropout](#sec_dropout).
    1. Viết hàm mục tiêu, phụ thuộc vào tất cả siêu tham số của mô hình và `batch_size`. Dùng `max_epochs=50`. GPU không giúp ích ở đây, nên dùng `num_gpus=0`. Gợi ý: Sửa `hpo_objective_lenet`.
    2. Chọn một không gian tìm kiếm hợp lý, trong đó `num_hiddens_1`, `num_hiddens_2` là các số nguyên trong $[8, 1024]$, và giá trị dropout nằm trong $[0, 0.95]$, còn `batch_size` nằm trong $[16, 384]`. Cung cấp code cho `config_space`, dùng các phân phối hợp lý từ `scipy.stats`.
    3. Chạy random search trên ví dụ này với `number_of_trials=20` và vẽ kết quả. Đảm bảo trước tiên đánh giá cấu hình mặc định của [sec_dropout](#sec_dropout), đó là `initial_config = {'num_hiddens_1': 256, 'num_hiddens_2': 256, 'dropout_1': 0.5, 'dropout_2': 0.5, 'lr': 0.1, 'batch_size': 256}`.
2. Trong bài tập này, bạn sẽ triển khai một searcher mới (subclass của `HPOSearcher`) đưa ra quyết định dựa trên dữ liệu quá khứ. Nó phụ thuộc vào các tham số `probab_local`, `num_init_random`. Phương thức `sample_configuration` của nó hoạt động như sau. Với `num_init_random` lời gọi đầu tiên, làm giống `RandomSearcher.sample_configuration`. Sau đó, với xác suất `1 - probab_local`, làm giống `RandomSearcher.sample_configuration`. Ngược lại, chọn cấu hình đã đạt lỗi validation nhỏ nhất cho đến nay, chọn ngẫu nhiên một siêu tham số của nó, và lấy mẫu giá trị của siêu tham số đó ngẫu nhiên như trong `RandomSearcher.sample_configuration`, nhưng giữ nguyên mọi giá trị khác. Trả về cấu hình này, giống hệt cấu hình tốt nhất cho đến nay, ngoại trừ một siêu tham số này.
    1. Viết code cho `LocalSearcher` mới này. Gợi ý: Searcher của bạn cần `config_space` làm đối số khi xây dựng. Bạn có thể dùng một member kiểu `RandomSearcher`. Bạn cũng sẽ phải triển khai phương thức `update`.
    2. Chạy lại thí nghiệm từ bài tập trước, nhưng dùng searcher mới của bạn thay vì `RandomSearcher`. Thử nghiệm với các giá trị khác nhau cho `probab_local`, `num_init_random`. Tuy nhiên, lưu ý rằng so sánh đúng giữa các phương pháp HPO khác nhau đòi hỏi lặp lại thí nghiệm nhiều lần, và lý tưởng là xét một số tác vụ benchmark.


[Thảo luận](https://discuss.d2l.ai/t/12092)
