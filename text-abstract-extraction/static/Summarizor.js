function connect() {
	        /*var host = "http://127.0.0.1:9999/get_summary?tt={}&ctt={}&ppt={}";
	        document.getElementById("Get-abstract").value = "生成摘要";
	        document.getElementById("Get-abstract").disabled = true;
            
            try {
	
	            socket.onopen = function (msg) {
	                document.getElementById("btnConnect").value = "生成摘要";	                
	                //alert("生成摘要！");
	            };*/
	
	        //get请求
            var data = {
                "tt": "{}",
                "ctt": "{}",
                "ppt": "{}"
            };
            $.ajax({
                type: 'GET',
                url: 'http://127.0.0.1:9998/get_summary',
                data: data,
                datatype: 'string',//希望服务器返回字符格式
                success： function(data){    
                },
                error: function(xhr,type){
                }
            });
            //post请求
            var data={}
            data['tt'] = $('#my-title');
            data['ctt'] = $('#my-input');
            data['ptt'] = $('.my-proportion');
            $.ajax({
                type: 'POST',
                url: 'http://127.0.0.1:9998/get_summary',
                data: data,
                dataType: 'json', //注意：这里是指希望服务端返回json格式的数据
                success: function(data) {    //这里的data就是json格式的数据
                },
                error: function(xhr, type) {
                }
            });
            
            