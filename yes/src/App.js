import React, { Component } from 'react'
import './App.css'
import Box from '@material-ui/core/Box';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import Dropzone from 'react-dropzone'
import styled from 'styled-components';
import CircularProgress from '@material-ui/core/CircularProgress';
import Slider from '@material-ui/core/Slider';
import Input from '@material-ui/core/Input';
import Alert from '@material-ui/lab/Alert';

const getColor = (props) => {
  if (props.isDragAccept) {
      return '#00e676';
  }
  if (props.isDragReject) {
      return '#ff1744';
  }
  if (props.isDragActive) {
      return '#2196f3';
  }
  return '#eeeeee';
}

const Container = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  border-width: 2px;
  border-radius: 2px;
  border-color: ${props => getColor(props)};
  border-style: dashed;
  background-color: #fafafa;
  color: #bdbdbd;
  outline: none;
  padding: 50px;
  margin:50px;
  transition: border .24s ease-in-out;
`;
class App extends Component {
  constructor (props) {
    super(props)
    this.state = {
      file: '',
      output: null,
      processed: false,
      processing: false,
      value: 40,
      audio: 1,
      color: 7,
      motion: 2,
      error: false
    }
  }
  
  componentDidMount(){
    try {
      var check = setInterval(async () => {
                  if((this.state.file != '') && this.state.processing){
                    fetch(`http://127.0.0.1:8000`+this.state.file, {
                      method: "GET",
                      headers: {
                        'Access-Control-Allow-Origin': '*'
                        }, 
                    }).then(response => response.blob()).then(blob => {   try{
                                                                              if(blob.type == "application/json"){
                                                                                console.log("Still Processing")
                                                                              } else if(blob.type == "image/png"){
                                                                                var url = window.URL.createObjectURL(blob);
                                                                                document.getElementById('ItemPreview').src = url
                                                                                clearInterval(check);
                                                                                this.setState({processed: true, processing: false})
                                                                              } else{
                                                                                throw new Error('Unknown File Type')
                                                                              }
                                                                            } catch(e){
                                                                              console.log(e)
                                                                            }
                                                                          
                                                                      })
                  }
                }, 5000);
    } catch(e) {
      console.log(e);
    }
  }

  handleSubmit(e){
    const formData = new FormData();
    const file = e[0]
    formData.append("file", file, file.name);
    formData.append("parameters", this.state.color.toString() + this.state.motion.toString() + this.state.audio.toString())
    fetch(`http://127.0.0.1:8000/uploadfiles/`, {
      method: "POST",
      body: formData,
      headers: {
        'Access-Control-Allow-Origin': '*'
        }, 
    }).then(response => response.json()).then(data => {this.setState({file: data['file_name'], processing: true})})
  }
  valuetext(value) {
    return `${value}°C`;
  }
  handleChange = (event, newValue, name) => {
    var oldValue = this.state[name]
    this.setState({[name]: newValue})
    if((this.state.color + this.state.motion + this.state.audio) > 10){
      this.setState({[name]: oldValue, error: true})
    }
  };
  render () {
    return (
      <div className="App">
      <Box>
        <Typography variant="h1" component="h2">Thum<span style={{color: '#3F51B5'}}>Bio</span></Typography >
        {this.state.processing && !this.state.processed && <div><Box mb={10}><Typography variant="p"  component="p">Your Video Is Being Processed</Typography ></Box><CircularProgress /></div>}
        {!this.state.processing && !this.state.processed &&
            <div>
              <div>
                <Box><Typography variant="p" component="p">Weight Configuration</Typography ></Box>
                <Typography variant="h6" component="h6">Color</Typography >
                <Slider
                  style={{'width': '50%'}}
                  defaultValue={this.state.color}
                  getAriaValueText={this.valuetext}
                  aria-labelledby="discrete-slider-small-steps"
                  step={1}
                  marks
                  min={0}
                  max={10}
                  value = {this.state.color}
                  id="color_value"
                  onChange={(e,value) => this.handleChange(e,value,'color')}
                  valueLabelDisplay="on"
                />
              </div>
              <div>
                <Typography variant="h6" component="h6">Motion</Typography >
                <Slider
                  style={{'width': '50%'}}
                  defaultValue={this.state.motion}
                  getAriaValueText={this.valuetext}
                  aria-labelledby="discrete-slider-small-steps"
                  step={1}
                  marks
                  min={0}
                  max={10}
                  value = {this.state.motion}
                  id="motion_value"
                  onChange={(e,value) => this.handleChange(e,value,'motion')}
                  valueLabelDisplay="on"
                />
              </div>
              <Box text-align='center' style={{'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'align-items': 'center'}}>
                <Typography variant="h6" component="h6">Audio</Typography >
                <Slider
                  style={{'width': '50%'}}
                  defaultValue={this.state.audio}
                  getAriaValueText={this.valuetext}
                  aria-labelledby="discrete-slider-small-steps"
                  step={1}
                  marks
                  min={0}
                  max={10}
                  value = {this.state.audio}
                  id="audio_value"
                  onChange={(e,value) => this.handleChange(e,value,'audio')}
                  valueLabelDisplay="on"
                />
                {this.state.error && <Alert style={{'width': '50%'}} severity="error">Your weights must add up to 10</Alert>}
              </Box>

            <Dropzone multiple={false} onDrop={acceptedFiles => this.handleSubmit(acceptedFiles)}>
            {({getRootProps, getInputProps, isDragActive, isDragAccept, isDragReject}) => (
              <Container {...getRootProps({isDragActive, isDragAccept, isDragReject})}>
                  <input {...getInputProps()} />
                  <p>Drag 'n' drop a video here! (Or click to select a video)</p>
              </Container>
            )}
          </Dropzone>
        </div>
        }
      {(this.state.processed) &&
      <Typography variant="h4" component="h2">Output</Typography >      
      }
            <img id="ItemPreview" src="" />      

      </Box>
      </div>
    )
  }
}


export default App