$(function(){
	var result = {};
	var oPop = $('#pop_con');
	var iTime = 108;
	oPop.css({display:'block'});
	
	
	$(".my-input").val('网易娱乐7月21日报道 林肯公园主唱查斯特·贝宁顿Chester Bennington于今天早上，在洛杉矶帕洛斯弗迪斯的一个私人庄园自缢身亡，年仅41岁。此消息已得到洛杉矶警方证实。洛杉矶警方透露，Chester的家人正在外地度假，Chester独自在家，上吊地点是家里的二楼。一说是一名音乐公司工作人员来家里找他时发现了尸体，也有人称是佣人最早发现其死亡。\
　　林肯公园另一位主唱麦克·信田确认了Chester Bennington自杀属实，并对此感到震惊和心痛，称稍后官方会发布声明。Chester昨天还在推特上转发了一条关于曼哈顿垃圾山的新闻。粉丝们纷纷在该推文下留言，不相信Chester已经走了。\
　　外媒猜测，Chester选择在7月20日自杀的原因跟他极其要好的朋友、Soundgarden(声音花园)乐队以及Audioslave乐队主唱Chris Cornell有关，因为7月20日是Chris Cornell的诞辰。而Chris Cornell于今年5月17日上吊自杀，享年52岁。Chris去世后，Chester还为他写下悼文。\
　　对于Chester的自杀，亲友表示震惊但不意外，因为Chester曾经透露过想自杀的念头，他曾表示自己童年时被虐待，导致他医生无法走出阴影，也导致他长期酗酒和嗑药来疗伤。目前，洛杉矶警方仍在调查Chester的死因。\
　　据悉，Chester与毒品和酒精斗争多年，年幼时期曾被成年男子性侵，导致常有轻生念头。Chester生前有过2段婚姻，育有6个孩子。\
　　林肯公园在今年五月发行了新专辑《多一丝曙光One More Light》，成为他们第五张登顶Billboard排行榜的专辑。而昨晚刚刚发布新单《Talking To Myself》MV。');
	$(".form-control").val('林肯公园林肯公园主唱查斯特·贝宁顿Chester Bennington自杀，年仅41。');
	$("input").eq(1).click(function(){
		$.ajax({
			url:'/AbastractGeneration/mysql',
			type:'GET',
			cache: false,
			dataType:'json',
			success:function(result){
				$("textarea").val(result['content']);
				$(".form-control").val(result['title']);
			}
		})
	})

	$("input").eq(3).click(function(){
		var text = $("textarea").val();
		var num = $("#len option:selected").text();
		var title = $(".form-control").val();
		var data = {'text':text, 'num':num, 'title':title}
		var req_json = JSON.stringify(data);

		$.ajax({
			url:'/AbastractGeneration/solve',
			type:'post',
			data: req_json,
			contentType: "application/json",
			dataType:'json',
		})
		.done(function(dat){
			result = dat;
			
			$(".nav-stacked > li").eq(0).click();
		})
		
	})

	$("input").eq(2).click(function(){
		$("textarea").val('');
		$(".form-control").val('');

	})



})