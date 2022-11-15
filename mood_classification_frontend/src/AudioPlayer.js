import React from "react";
import "../src/assets/css/audio.scss"
import "./assets/css/ProfileCard.css"
import {faPlay,faStop,faPause} from "@fortawesome/free-solid-svg-icons";
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";
import TextField from '@material-ui/core/TextField';
import Autocomplete, { createFilterOptions } from '@material-ui/lab/Autocomplete';
import DetailCard from "./detailCard";

import tol from './assets/songs/TOL.mp3'

import 'react-select-search/style.css'
export default class AudioPlayer extends React.Component {
    constructor() {
        super();
        this.state = {
            songId:'',
            songTitles : [
                "Thinking out loud",
                "Send My Love (To Your New Lover)",
                "Gangsta"
            ],
            selected_title:"",
            recommendation1: {
                artistName: "Alessia Cara", trackName: "How Far I'll Go",artworkUrl100: "https://images.genius.com/2309f6cbf6dec566ecb4aad05b32cd3e.500x500x1.jpg"
            },
            recommendation2: {
                artistName: "Imagine Dragons ", trackName: "Thunder",artworkUrl100: "https://upload.wikimedia.org/wikipedia/en/2/28/Imagine_Dragons_Thunder.jpg"
            },

            song: {
                valuesongid:12,artistName: "Adele", trackName: "Send My Love (To Your New Lover)", valence: "Positive", arousal:"Positive",avatar:'https://cdn.pixabay.com/photo/2020/12/27/20/24/smile-5865208_1280.png',
                previewUrl: "https://opradre.com/wp-content/uploads/2022/05/Adele-Send-My-Love.mp3", artworkUrl100: "https://upload.wikimedia.org/wikipedia/en/9/96/Adele_-_25_%28Official_Album_Cover%29.png", emotion:"Happy"},
            playing: undefined,
            imageClicked: false,
            songList : [
                {
                    valuesongid:12,artistName: "Adele", name: "Send My Love (To Your New Lover)", valence: "Positive", arousal:"Positive",
                    previewUrl: "https://opradre.com/wp-content/uploads/2022/05/Adele-Send-My-Love.mp3", artworkUrl100: "/Users/rashini/mood_classification/src/assets/images/adele_artwork.png", emotion:"Happy"
                } ,
                {
                    value:2,artistName: "Kehlani", name: "Gangsta", valence: "Negative", arousal:"Negative",
                    previewUrl: "./assets/songs/506.mp3", artworkUrl100: "https://upload.wikimedia.org/wikipedia/en/5/52/GangstaKehlani.jpeg"
                } ,
                {
                    value:1,artistName: "Alessia Cara", name: "How Far I'll Go(Alessia Cara Version)", valence: "Positive", arousal:"Positive",
                    previewUrl: "ZNra8eK0K6k.webm", artworkUrl100: "https://is2-ssl.mzstatic.com/image/thumb/Music111/v4/84/60/76/84607655-c632-5391-67ca-7cc89b23cfbc/source/100x100bb.jpg"
                } ,
                {
                    value:4, artistName: "Alessia Cara", name: "How Far I'll Go(Alessia Cara Version)", valence: "Positive", arousal:"Positive",
                    previewUrl: "https://mdundo.com/song/1195402#", artworkUrl100: "https://is2-ssl.mzstatic.com/image/thumb/Music111/v4/84/60/76/84607655-c632-5391-67ca-7cc89b23cfbc/source/100x100bb.jpg"
                } ,

            ]
        }
        this.searched = this.searched.bind(this)
    }


    filterOptions = createFilterOptions({
        matchFrom: 'start',
        stringify: option => option,
    }
    );

    imageClick = (e) => {
        const audio = document.getElementById('player')
        this.setState({
            imageClicked: !this.state.imageClicked
        })
        if(this.state.imageClicked) {
            this.setState({
                playing: true
            })
            audio.play()
        } else{
            this.setState({
                playing: false
            })
            audio.pause()
        }
    }

    searched(e){
        console.log("aawa")

        this.setState({ selected_title: e.target.value });
        console.log(this.state.selected_title)

        if (this.state.selected_title === 'Thinking out loud'){
            this.setState({
                song:{
                    value:2, artistName: "Ed Sheeran", trackName: "Thinking out loud", valence: "Positive", arousal:"Negative", emotion:'Calm', avatar:'https://i.pinimg.com/originals/92/18/2e/92182edcc049d431cb62ce2b2527a148.png',
                    previewUrl: "./assets/songs/506.mp3", artworkUrl100: "https://i.scdn.co/image/ab67616d0000b27313b3e37318a0c247b550bccd"},
                recommendation1: {
                    artistName: "John Mayer", trackName: "Moving on and Getting over",artworkUrl100: "https://i.scdn.co/image/ab67616d0000b273c6bfaf942ed981d5c4c922e4"
                },
                recommendation2: {
                    artistName: "Wale ", trackName: "My PYT",artworkUrl100: "https://i.scdn.co/image/ab67616d0000b27345f98968abd4d0a66a69fc46"
                },
            })
        }

    }

    render(){
        const song = this.state.song
        return(

            <div>
                <div style={{ marginLeft: '40%', marginTop: '150px' }}>
                    <h3>Find your song!</h3>
                    <Autocomplete
                        style={{ width: 500 }}
                        freeSolo
                        filterOptions={this.filterOptions}
                        options={this.state.songTitles}
                        renderInput={(params) => (
                            <TextField {...params}
                                       variant="outlined"
                                       label="Search Box"

                                       onChange={this.searched}
                            />
                        )}
                    />
                </div>
            <div class="frame" style={{
                position: 'absolute',
                left: '150px',
                top:'400px',
                // width: '200px'
            }}>
                <div class="center">
                    <div class="main">
                        <div class={this.state.playing ? 'artwork-cover play-artwork-cover': 'artwork-cover'}>
                            <img src= {this.state.song.artworkUrl100} style={{height:'100px',objectFit: 'cover'}} onClick={this.imageClick} class={this.state.playing ? 'play-artwork': 'pause-artwork'}/>
                        </div>
                        <audio id="player" src={tol} loop />
                    </div>

                    <div class="right">
                        <div class="player-controls">
                            <div class="player-control" onClick={this.imageClick}>
                                <FontAwesomeIcon icon={faPlay} color="teal" size="2x"/>
                            </div>
                            <div class="player-control" onClick={this.imageClick}>
                                <FontAwesomeIcon icon={faPause} color="teal" size="2x"/>

                            </div>
                            <div class="player-control" onClick={this.imageClick}>
                                <FontAwesomeIcon icon={faStop} color="teal" size="2x"/>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
                <div style={{
                    position: 'absolute',
                    left: '550px',
                    top:'400px',
                    width: '200px'}}>
                    <DetailCard
                        name={this.state.song.trackName}
                        emotion={this.state.song.emotion}
                        arousal={this.state.song.arousal}
                        valence={this.state.song.valence}
                        avatar={this.state.song.avatar}
                        songTitle={this.state.song.trackName}
                        artist={this.state.song.artistName}
                    />
                </div>
                <div>


                <div style={{
                    position: 'absolute',
                    left: '1250px',
                    top:'200px',
                    width: '200px'}}>
                    <h2>RECOMMENDATIONS</h2>
                    <div style={{
                        boxShadow: '0 4px 8px 0 rgba(0,0,0,0.2)',
                        height: '280px',
                        width: '250px',
                        position: 'relative',
                        marginBottom: '10%'
                    }}>
                        <img style={{ width: '100%',
                            height: '100%'}}
                            src={this.state.recommendation1.artworkUrl100}
                            alt=""
                        />
                        <div style={{
                            bottom: '0',
                            zIndex: '9',
                            position: 'absolute',
                            backgroundColor: 'rgba(255, 255, 255, 0.7)',
                            display: 'flex',
                            flexDirection: 'column',
                            width: '100%',
                            alignItems: 'center',
                            height: '35%',
                            paddingBottom: '20px',
                            marginTop:'-50px'


                        }}>
                            <h2 style={{textAlign: 'center',
                                fontSize: '1.5rem'}}>{this.state.recommendation1.trackName}</h2>
                            <span> {this.state.recommendation1.artistName}</span>
                        </div>
                    </div>
                    <div style={{
                        boxShadow: '0 4px 8px 0 rgba(0,0,0,0.2)',
                        height: '280px',
                        width: '250px',
                        position: 'relative',
                        marginBottom: '10%'
                    }}>
                        <img style={{ width: '100%',
                            height: '100%'}}
                             src={this.state.recommendation2.artworkUrl100}
                             alt=""
                        />
                        <div style={{
                            bottom: '0',
                            zIndex: '9',
                            position: 'absolute',
                            backgroundColor: 'rgba(255, 255, 255, 0.7)',
                            display: 'flex',
                            flexDirection: 'column',
                            width: '100%',
                            alignItems: 'center',
                            height: '35%',
                            paddingBottom: '20px',


                        }}>
                            <h2 style={{textAlign: 'center',
                                fontSize: '1.5rem'}}>{this.state.recommendation2.trackName}</h2>
                            <span>{this.state.recommendation2.artistName}</span>
                        </div>
                    </div>
                </div>
                </div>
            </div>
        )
    }
}

