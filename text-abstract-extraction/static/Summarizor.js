function connect()
{
  var tt = document.getElementById("my-title").value;
  var ctt = document.getElementById("my-input").value;
  var ppt = document.getElementById("my-proportion").value;
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
xmlhttp.open("GET","http://127.0.0.1:9999/get_summary?"+str,true);
xmlhttp.send();


}

$("#Get-abstract").click(connect);