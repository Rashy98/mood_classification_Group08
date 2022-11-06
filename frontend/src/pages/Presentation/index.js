// @mui material components
import Container from "@mui/material/Container";
import Card from "@mui/material/Card";

// Material Kit 2 React components
import MKBox from "components/MKBox";

// Images
import bgImage from "assets/images/bg-1.png";
import CardProfile from "pages/Presentation/components/BuiltByDevelopers";
import ExampleCard from './components/ExampleCard'

const divStyle = {
    display: 'flex',
    alignItems: 'center',
    marginLeft:'5vh'
};

function Presentation() {
    return (
        <>
            <MKBox
                minHeight="40vh"
                width="100%"
                sx={{
                    backgroundImage: `url(${bgImage})`,
                    backgroundSize: "cover",
                    backgroundPosition: "top",
                    display: "grid",
                    placeItems: "center",
                }}
            />
            <Card
                sx={{
                    p: 2,
                    mx: { xs: 2, lg: 3 },
                    mt: -8,
                    mb: 4,
                    backgroundColor: ({ palette: { white }, functions: { rgba } }) => rgba(white.main, 0.8),
                    backdropFilter: "saturate(200%) blur(30px)",
                    boxShadow: ({ boxShadows: { xxl } }) => xxl,
                }}
            >
                <Container sx={{ mt: 3}}>
                    <div style={divStyle}>
                        <CardProfile />
                        <div>
                            <div>
                                <ExampleCard
                                    name="song"
                                    emotion="Happy"
                                    arousal="Positive"
                                    valence="Positive"
                                    songTitle="Made you look"
                                    artist="Meghan Trainor"
                                />
                            </div>
                        </div>



                    </div>

                </Container>

            </Card>

        </>
    );
}

export default Presentation;
