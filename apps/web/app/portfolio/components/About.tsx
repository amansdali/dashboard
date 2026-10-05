export default function About(){
    return (
        <section id="about-section">
            <div style={{ height: '15rem' }}></div>
            <h2 className="header-2">about me</h2>
            <p className="body-small">
                Welcome to my portfolio website!<br/><br/>
                I am a third year student at the University of Toronto, pursuing a computer science specialist
                and statistics major, with focuses in Artificial Intelligence and Game Design.<br/><br/>
                I am passionate about learning new skills to bring innovative ideas to life, combining logic and
                creativity to build software that is interesting, useful, and fun.<br/><br/>
                Continue scrolling down to learn more about my skills, projects, and contact information.
            </p>
            <div className="small-arrow-divider">
                <a href="#skills-section" aria-label="Navigate to Skills Section">
                    <svg className="small-arrow" viewBox="0 0 57 54" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <line x1="4.24264" y1="4" x2="29" y2="28.7574" strokeWidth="6" strokeLinecap="round"/>
                        <line x1="3" y1="-3" x2="38.0122" y2="-3"
                              transform="matrix(0.707107 -0.707107 -0.707107 -0.707107 25 29)" strokeWidth="6"
                              strokeLinecap="round"/>
                        <line x1="4.24264" y1="25" x2="29" y2="49.7574" strokeWidth="6" strokeLinecap="round"/>
                        <line x1="3" y1="-3" x2="38.0122" y2="-3"
                              transform="matrix(0.707107 -0.707107 -0.707107 -0.707107 25 50)" strokeWidth="6"
                              strokeLinecap="round"/>
                    </svg>
                </a>
            </div>
        </section>
    )
}