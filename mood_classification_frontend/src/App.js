import logo from './logo.svg';
import './App.css';
import AudioPlayer from "./AudioPlayer";
import React from "react";
import SearchBarNew from "./searchbar";
import Carousel from 'react-bootstrap/Carousel';
import bg from "./assets/images/bg-1.png"
import DetailCard from "./detailCard";
import Searchbar from "./search";
const divStyle = {
    display: 'flex',
    alignItems: 'center',
    marginLeft:'5vh',
};

function App() {
  return (
      <div>
          {/*<Searchbar />*/}
    <div className={divStyle}>
        <div >
            <AudioPlayer />
        </div>

    {/*    style={{*/}
    {/*    position: 'absolute',*/}
    {/*    left: '150px',*/}
    {/*    top:'300px',*/}
    {/*    // width: '200px'*/}
    {/*}}*/}


        {/*<div style={{*/}
        {/*    position: 'absolute',*/}
        {/*    right: '650px',*/}
        {/*    top:'250px',*/}
        {/*    width: '200px'}}>*/}
        {/*    <DetailCard*/}
        {/*        name="song"*/}
        {/*        emotion="Happy"*/}
        {/*        arousal="Positive"*/}
        {/*        valence="Positive"*/}
        {/*        songTitle="Made you look"*/}
        {/*        artist="Meghan Trainor"*/}
        {/*    />*/}
        {/*</div>*/}

      {/*<SearchBarNew />*/}

    </div>
      </div>
  );
}

export default App;
