/***************************************
//ctx: Canvas 2D context
//data: json 数据
//r: 半径
****************************************/
function piechart(ctx, data, r) {

    this.ctx = ctx;

    //Json 数据
    this.data = data;

    //半径
    this.r = r;    
};
 
//画图
piechart.prototype.draw = function () {

    //饼图标题
    var title = this.data.title;
    //饼图每部分颜色
    var colors = [ "#99CC33",  "#FF6666", "#CCCCFF", "#999933", "#009966"];
    
    //数据集合
    var dataarray = this.data.data;

    //计算画布的尺寸
    this.ctx.canvas.height = this.r * 2 + 120; 
    this.ctx.canvas.width = this.r * 2 + 220;

    //绘制标题
    this.ctx.font = "bold 16px Arial";
    this.ctx.textAlign = "center";
    this.ctx.fillText(title, this.ctx.canvas.width / 2, 25);

    //圆心
    var circlecenter = { "x": this.r + 50, "y": this.r + 70 };
    
    //顺时针画圆弧， 记录圆弧画到的位置
    var lastendPI = 0 * Math.PI;

    //用于记录画了多少的圆弧， 占圆的比例
    var angle = parseFloat("0.0");

    //圆弧开始端的坐标
    var arc_oneend = { "x": circlecenter.x + this.r, "y": circlecenter.y };

    //圆弧结束端的坐标
    var arc_anotherend = { "x": circlecenter.x + this.r, "y": circlecenter.y };
   
    //标识线开始端的坐标
    var remark_start = { "x": 0, "y": 0 };

    //标识线结束端的坐标
    var remark_end = { "x":0, "y":0 };

    //循环每组数据
    for (var i = 0; i < dataarray.length; i++) {
      
        arc_oneend.x = arc_anotherend.x;
        arc_oneend.y = arc_anotherend.y;

        //计算标识线开始端的坐标
        remark_start.x = circlecenter.x + this.r * Math.cos((angle + parseFloat(dataarray[i].persent) / 2) * 2 * Math.PI);
        remark_start.y = circlecenter.y + this.r * Math.sin((angle + parseFloat(dataarray[i].persent) / 2) * 2 * Math.PI);

        //累加所画圆弧占圆的比例
        angle += parseFloat(dataarray[i].persent);
        
        //计算圆弧结束端的坐标
        arc_anotherend.x = circlecenter.x + this.r * Math.cos(angle * 2 * Math.PI);
        arc_anotherend.y = circlecenter.y + this.r * Math.sin(angle * 2 * Math.PI);

        ////计算标识线结束端的坐标
        remark_end.x = remark_start.x > circlecenter.x ? remark_start.x + 20 : remark_start.x - 20;
        remark_end.y = remark_start.y > circlecenter.y ? remark_start.y + 20 : remark_start.y - 20;

         //绘制标识线
        this.ctx.beginPath();
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = "gray";
        this.ctx.moveTo(remark_start.x, remark_start.y);
        this.ctx.lineTo(remark_end.x, remark_end.y);
        this.ctx.stroke();

        //绘制比例数字
        this.ctx.font = "12px Arial";
        this.ctx.fillStyle = "black";
        this.ctx.fillText(dataarray[i].persent * 100 + "%", remark_end.x, remark_end.y > circlecenter.y ? remark_end.y + 10 : remark_end.y);
       
        //画出扇形
        this.ctx.beginPath();
        this.ctx.lineWidth = 1;
        this.ctx.moveTo(arc_oneend.x ,arc_oneend.y);
        this.ctx.lineTo(circlecenter.x, circlecenter.y);
        this.ctx.lineTo(arc_anotherend.x, arc_anotherend.y);
        this.ctx.arc(circlecenter.x, circlecenter.y, this.r, lastendPI, lastendPI + dataarray[i].persent * 2 * Math.PI);
        
        //扇形填充颜色
        this.ctx.strokeStyle = colors[i];
        this.ctx.fillStyle = colors[i];
        this.ctx.fill();
        this.ctx.stroke();

        //记录圆弧画到的位置，作为下一数据说对应弧形的起点
        lastendPI += dataarray[i].persent * 2 * Math.PI;       
    }

    //绘制标识器
    this.ctx.lineWidth = 15;
    for (var i = 0; i < dataarray.length; i++) {

        this.ctx.beginPath();
        this.ctx.strokeStyle = colors[i];
        this.ctx.fillStyle = "black";
        this.ctx.moveTo(this.ctx.canvas.width - 80, 50 + i * 17);
        this.ctx.lineTo(this.ctx.canvas.width - 65, 50 + i * 17);
        this.ctx.font = "12px Arial";
        this.ctx.textAlign = "left";
        this.ctx.fillText(dataarray[i].title, this.ctx.canvas.width - 60, 50 + i * 17 + 4);
        this.ctx.stroke();
    }
}

 