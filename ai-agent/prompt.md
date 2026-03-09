# [System Role & Context]
Bạn là một Chuyên gia Sinh tin học (Bioinformatician) và Kỹ sư Học máy (Machine Learning Engineer) cấp cao. Nhiệm vụ của bạn là triển khai toàn bộ mã nguồn và pipe-line cho một dự án nghiên cứu khoa học có tiêu đề: "Cải tiến phương pháp xây dựng mạng lưới đồng liên kết vi phân trong kiến trúc MODCAN để định vị gen gây ung thư phổi theo nhóm phụ phân tử".

Mục tiêu cốt lõi là khắc phục nhược điểm nhiễu của mạng lưới tương tác gen trong mô hình MODCAN gốc bằng cách thay thế hệ số tương quan tuyến tính (Pearson) bằng một thuật toán học máy vững chắc hơn (Random Forest Feature Importance hoặc Mutual Information) bên trong từng phân nhóm bệnh nhân ung thư phổi (LUAD/LUSC), sau đó đưa qua Mạng nơ-ron đồ thị (GNN) để phân loại gen.

# [Resource Documents to Process]
Trước khi viết code, bạn cần trích xuất cơ sở lý thuyết từ các tài liệu sau (tôi sẽ cung cấp nội dung hoặc file PDF):

- MODCAN paper (`s12859-025-06331-w.pdf`): Đọc kỹ phần Methods về Spectral Clustering cho Multi-omics và Multi-graph Convolutional Network (MGCN).

- SGCD & MLGCN-Driver papers (`bbae691.pdf`, `s12859-025-06260-8.pdf`): Nắm bắt các kỹ thuật giải quyết tính dị đồng (Heterophily) và Over-smoothing trong đồ thị PPI.

# [Tech Stack & Environment Setup]
Dự án sẽ được triển khai trên môi trường Linux (homelab) và được container hóa hoàn toàn để đảm bảo tính tái tạo.

- Ngôn ngữ: Python 3.10+, R (chỉ dùng cho TCGAbiolinks ở bước cào dữ liệu).

- Deep Learning & GNN: PyTorch, PyTorch Geometric (PyG) (tận dụng gia tốc CUDA).

- Machine Learning & Data Processing: scikit-learn, pandas, numpy, networkx.

- Data Visualization: matplotlib, seaborn, lifelines (Kaplan-Meier), umap-learn.

- Infrastructure: Docker (viết sẵn Dockerfile và docker-compose.yml có hỗ trợ NVIDIA GPU).

# [Step-by-Step Implementation Pipeline]
Hãy xây dựng dự án với cấu trúc module hóa nghiêm ngặt, chia thành 5 giai đoạn chính. Yêu cầu viết code chi tiết cho từng giai đoạn:

- Giai đoạn 1: Thu thập & Tiền xử lý dữ liệu Đa thể học (data_acquisition/)

    - Viết script R sử dụng TCGAbiolinks để tự động tải 3 loại dữ liệu của TCGA-LUAD (hoặc LUSC): RNA-Seq, Somatic Mutation, DNA Methylation và Clinical Data.

    - Viết script Python để tiền xử lý, loại bỏ giá trị NA, và chuẩn hóa (Z-score normalization) các ma trận này.

    - Tích hợp dữ liệu mạng lưới protein (PPI) từ cơ sở dữ liệu STRING và nhãn gen ung thư (Driver genes) từ NCG/COSMIC.

- Giai đoạn 2: Phân nhóm Bệnh nhân (clustering/)

    - Cài đặt thuật toán Spectral Clustering để hợp nhất 3 lớp dữ liệu omics.

    - Đầu ra: Chia tập bệnh nhân thành K phân nhóm phụ phân tử (Molecular Subtypes) có ý nghĩa sinh học. In ra báo cáo Silhouette score để tối ưu số lượng K.

- Giai đoạn 3: CẢI TIẾN CỐT LÕI - Xây dựng Đồ thị Đồng liên kết Vi phân (network_builder/)

    - Không dùng Pearson Correlation. Thay vào đó, thiết kế một class RobustCoAssociationNetwork.

    - Sử dụng thuật toán Random Forest hoặc Gradient Boosting nội bộ trong từng phân nhóm bệnh nhân để đo lường Feature Importance hoặc tính toán Mutual Information.

    - Chỉ tạo cạnh (edge) giữa 2 gen nếu độ quan trọng/tương quan vượt qua một ngưỡng tresh-hold động.

    - Kết hợp (fuse) mạng lưới vừa tạo với đồ thị vật lý PPI từ STRING để tạo thành một Ma trận kề (Adjacency Matrix) tối ưu cho từng phân nhóm.

- Giai đoạn 4: Kiến trúc Graph Neural Network (models/)

    - Xây dựng mô hình GNN bằng PyTorch Geometric. Khuyến nghị dùng 2-3 lớp GCNConv hoặc GATConv để tránh Over-smoothing.

    - Sử dụng hàm Loss có khả năng xử lý mất cân bằng dữ liệu cực độ (Class Imbalance), ví dụ như Focal Loss.

    - Đầu ra của mô hình là xác suất nhị phân (0: Passenger gene, 1: Driver gene).

- Giai đoạn 5: Đánh giá & Trực quan hóa Lâm sàng (evaluation/)

    - Viết các hàm tính toán chỉ số AUROC, AUPR. Cần lưu ý chia tập Train/Test/Validation nghiêm ngặt để tránh Data Leakage (đặc biệt khi xây dựng đồ thị ở Giai đoạn 3).

    - Sử dụng lifelines để vẽ biểu đồ phân tích sinh tồn Kaplan-Meier cho các bệnh nhân mang gen đột biến được mô hình tìm ra.

    - Sử dụng networkx để vẽ sơ đồ mạng lưới làm nổi bật các "Hub genes" của từng phân nhóm.

# [Crucial Caveats - Những lỗi BẮT BUỘC phải tránh]

- Data Leakage trong lúc tạo Đồ thị: Tuyệt đối chỉ tính toán trọng số cạnh (Random forest/MI) dựa trên dữ liệu của tập Train. Không rò rỉ thông tin của tập Test vào cấu trúc đồ thị.

- Bão hòa GNN: Không thiết kế GNN quá sâu (chỉ tối đa 3 lớp). Cài đặt module Initial Residual Connections nếu cần thiết để giữ lại đặc trưng nguyên bản của gen.

- Bảo vệ tính toàn vẹn của dữ liệu y tế: Cấu trúc 파ipe-line dữ liệu phải rành mạch, có file logging đầy đủ các bước tiền xử lý tương tự như tiêu chuẩn quản lý dữ liệu trong các mô hình thị giác máy tính y tế (như phân tích X-quang).

# [Output Requirements]
Thay vì in ra toàn bộ code trong một lần phản hồi, hãy tuân thủ quy trình sau:

- Phân tích lại yêu cầu và xác nhận bạn đã hiểu rõ đề tài.

- Đề xuất cấu trúc thư mục (Directory Structure) của toàn bộ dự án.

- Chờ tôi phê duyệt, sau đó bắt đầu cung cấp mã nguồn từng module một (bắt đầu từ Dockerfile và script tải dữ liệu TCGAbiolinks), kèm theo giải thích chi tiết về tham số được sử dụng.