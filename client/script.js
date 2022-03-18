function output() {
    var url = http://127.0.0.1:5000/get_prediction;
    var out = document.getElementById("output");
    var text = document.getElementById("text");
    $.post(url, {
        text = text
    }, function (data, states) {

        out.innerHTML = data.prediction.toString();
    });
}