// https://app.zingsoft.com/demos/create/5WQIGCN8?fork 
var myConfig = {
  "type":"line",
  "plot":{
    "error":{
      'size': "8px",
      'line-color': "black",
      'line-width':1.45,
      'line-style': "solid"
    },
    "line-color": "#000080",
    "line-width": 1,
    "aspect":"spline",
    "stacked":false,
    "alpha-area":0.7,
    'legend-marker': {
      'show-line': true,
    }
  },
  "scale-x":{
    "offset-start":15,
    "offset-end":15,
    "guide":{
      visible: true,
      "line-style":"dashed",
      'alpha':0.1,
      'lineColor':'gray'
    },
    labels: ["10", "20", "30"],
    label: {
      text: 'Train Budget %',
      'font-family': "serif",
      'font-color': "black",
      'font-size':20
    },
    item: {
      'font-size':20,
      'font-family': "serif",
      'font-color': "black",
      'fontAngle': 0,
      offsetX: '20px'
    }
  },
  "scale-y":{
    "values":"35:70:5",
    "guide":{
      "line-style":"dashed",
      'alpha':0.6,
      'lineColor':'gray'
    },
    label: {
      text: 'Accuracy',
      'font-family': "serif",
      'font-color': "black",
      'font-size':20
    },
    item: {
      'font-size':20,
      'font-family': "serif",
      'font-color': "black"
    }
  },
  legend: {
    layout: "2x2",
    x: "40%",
    y: "60%",
    item: {
      'fontSize': 15,
      'font-family': "serif",
      'font-color': "black"
    },
    marker: {
        'show-line': true
    }
  },
    plotarea: {
    'margin-top': "1.5%",
    'margin-bottom': "23.3%",
    'margin-left': "9.5%",
    'margin-right': "3%",
  },
  "series":[
    {
      "values":[61.4386, 63.1754, 65.8596],
      "text": 'Classwise Min Var.',
      "line-color":"#black",
      "background-color":"#006600",
      "errors":[[0], [0], [0]],
      "marker":{
      "type":"circle",
      "size":5.5,
      "background-color":"#4472C4",
      "border-color":"#4472C4"
    },
    'legend-marker': {
        type: "circle",
        'background-color': "#4472C4",
        'border-color': "#4472C4",
        'border-width':1,
      }
    },
  {
      "values":[38.7895, 53.9474, 60.3684],
      "line-color":"#black",
      text: "Classwise Max Var.",
      "background-color":"#CC0000",
      "errors":[[0], [0], [0]],
      "marker":{
      "type":"triangle",
      "size":5.5,
      "background-color":"#00FF00",
      "border-color":"#00FF00"
    },
    'legend-marker': {
        type: "triangle",
        'background-color': "#00FF00",
        'border-color': "#00FF00",
        'border-width':1
      },
    },
    {
      "values":[44.5088, 54.4737, 60.9825],
      "line-color":"#black",
      text: "Classwise Min Conf.",
      "background-color":"#CC0000",
      "errors":[[0], [0], [0]],
      "marker":{
      "type":"square",
      "size":5.5,
      "background-color":"#ED7D31",
      "border-color":"#ED7D31"
    },
    'legend-marker': {
        type: "square",
        'background-color': "#ED7D31",
        'border-color': "#ED7D31",
        'border-width':1
      },
    },
    {
      "values":[61.1053, 65.6316, 66.9298],
      "line-color":"#black",
      text: "Classwise Max Conf.",
      "background-color":"#CC0000",
      "errors":[[0], [0], [0]],
      "marker":{
      "type":"star5",
      "size":5.5,
      "background-color":"#A020F0",
      "border-color":"#A020F0"
    },
    'legend-marker': {
        type: "star5",
        'background-color': "#A020F0",
        'border-color': "#A020F0",
        'border-width':1
      },
    }
  ]
};
zingchart.render({
    id : 'myChart',
    data : myConfig,
    height: 450,
    width: 650
});