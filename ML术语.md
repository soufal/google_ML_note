### ML术语：  

* Labels（标签）：指需要预测的真实事物$y$。（基本线性回归中的y变量）
* Features（特征）：表示数据的方式，用于描述数据的输入变量$x_i$。  
* Sample（样本）：数据的特定实例（为一个矢量）**$x$**。  
  * 有标签：具有{特征，标签}用于训练模型。
  * 无标签：具有{特征，？}用于对新数据做出预测。
* Model（模型）：将无标签样本映射到预测标签$y'$。  也就是执行预测的工具。样本输入到模型，模型通过训练数据来进行学习数据的规律，通过规律来预测无标签样本的标签。
* Loss（误差）：针对单个样本，模型的预测结果（$y'$)与真实值($y$)之间的方差。
* $L2$损失（平方误差）：预测值和真实值（观察值）之差的平方。
* 泛化（generalization）：模型很好地拟合以前从未见过的新数据的能力。也就是说对测试数据的预测能力。

