var myConfig = {
    "type":"line",
    "plot":{
      "error":{
        'size': "8px",
        'line-color': "black",
        'line-width':1.45,
        'line-style': "solid"
      },
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
      labels: ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"],
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
      "values":"60:72:1",
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
      layout: "1x2",
      x: "30%",
      y: "64%",
      item: {
        'fontSize': 20,
        'font-family': "serif",
        'font-color': "black"
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
        "values":[61.532, 64.4678, 66.77776, 68.8187, 68.8421, 69.3099, 70.508, 69.5146, 70.7836, 71.105],
        "text": 'Classwise Random',
        "line-color":"#black",
        "background-color":"#006600",
        "errors":[[0.9019], [0.99668], [0.594], [0.3317], [0.6204], [0.3844], [0.58025], [0.2111], [0.5255], [0.02909611662], [0.27289]],
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
        "values":[61.1696, 64.5965, 66.59, 67.1345, 67.6374, 68.5438, 69.807, 70.0818, 70.0526, 70.7193],
        "line-color":"#black",
        text: "Global Random",
        "background-color":"#CC0000",
        "errors":[[0.5796], [0.2305], [0.194], [0.4223], [0.4167], [0.1516], [0.4595], [0.322], [0.3485], [0]],
        "marker":{
        "type":"triangle",
        "size":5.5,
        "background-color":"#ED7D31",
        "border-color":"#ED7D31"
      },
      'legend-marker': {
          type: "triangle",
          'background-color': "#ED7D31",
          'border-color': "#ED7D31",
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