import React from "react";
import "./assets/css/ProfileCard.css";
import avatar from "./assets/images/happy.png";

function DetailCard(props) {
    return (
        <div className="card-container">
            <header>
                <img className="img_p" src={props.avatar} alt={props.name} />
            </header>
            <h1 className="bold-text">
                {props.emotion}
            </h1>
            <h2 className="normal-text">Arousal - {props.arousal}</h2>
            <h2 className="normal-text">Valence - {props.valence}</h2>
            <div className="social-container">
                <div className="followers">
                    <h1 className="bold-text">{props.songTitle}</h1>
                    <h2 className="smaller-text">Song title</h2>
                </div>
                <div className="photos">
                    <h1 className="bold-text">{props.artist}</h1>
                    <h2 className="smaller-text">Artist</h2>
                </div>
            </div>
        </div>
    );
}

export default DetailCard;
