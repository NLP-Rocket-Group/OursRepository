function connect()
{
  var tt = document.getElementById("my-title").value;
  if(tt =="") {
    tt = "摘要如下";
  }
  var ctt = document.getElementById("my-input").value;
  var ppt = document.getElementById("my-proportion").value;
    if(ppt == "") {
    ppt = "0.3";
  }
  var str = "tt="+tt+"&ctt="+ctt+"&ppt="+ppt;
  var xmlhttp;

  if (window.XMLHttpRequest)
  {
    // IE7+, Firefox, Chrome, Opera, Safari 浏览器执行代码
    xmlhttp=new XMLHttpRequest();
  }
  else
  {
    // IE6, IE5 浏览器执行代码
    xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
  }
  xmlhttp.onreadystatechange=function()
  {
    if (xmlhttp.readyState==4 && xmlhttp.status==200)
    {
    document.getElementById("my-output").innerHTML=xmlhttp.responseText;
    }
}

xmlhttp.open("GET","get_summary?"+str,true);
xmlhttp.send();


}

$("#Get-abstract").click(connect);