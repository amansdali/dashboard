import Image from 'next/image'

export default function Projects() {
    const projects = [
        {
            id: 'project1',
            href: 'https://github.com/amansdali/dashboard',
            image: '/images/websiteimg.png',
            title: 'Personal Website (WIP)',
            description: 'A personal website built using Next.js, featuring a portfolio page with an AI chatbot.',
            tags: ['Next.js', 'TypeScript', 'React', 'FastAPI', 'Python', 'OpenRouter', 'PostgreSQL'],
        },
        {
            id: 'project2',
            href: 'https://github.com/amansdali/Munchables',
            image: '/images/muchablesimg.png',
            title: 'Munchables',
            description: 'Social media app for foodies. Sign up, post recipes, join clubs, follow users, and scroll through a curated feed.',
            tags: ['Java', 'Java Swing', 'Github', 'Clean Architechture', 'Team Project'],
        },
        {
            id: 'project3',
            href: 'https://github.com/amansdali/Steam-Game-Recommendation-Tool',
            image: '/images/steamimg.png',
            title: 'Steam Game Recommendation Tool',
            description: 'Website that recommends games on Steam based on an algorithm using weighted graphs and trees.',
            tags: ['Python', 'Streamlit', 'HTML/CSS', 'Steam API','Team Project', 'Data Structures'],
        },
        {
            id: 'project4',
            href: 'https://github.com/amansdali/Exploding-Kittens-Local-Online-Game',
            image: '/images/kittensimg.png',
            title: 'Exploding Kittens Local Online Game',
            description: 'Multiplayer card game based on Exploding Kittens, made using Java. Features networking between four different devices.',
            tags: ['Java', 'Networking', 'Local Server', 'Multiplayer', 'Team Project'],
        },
        {
            id: 'project5',
            href: 'https://github.com/amansdali/The-Haunted-Forest',
            image: '/images/forestimg.png',
            title: 'The Haunted Forest',
            description: 'Platformer game made using Java for a high school project.',
            tags: ['Java', 'Game Dev', 'OOP', 'Solo Project'],
        },
        {
            id: 'project6',
            href: 'https://github.com/qutch/StoryOfMyLife',
            image: '/images/hackathonimg.png',
            title: 'Story Of My Life',
            description: 'A 2D top-down immersive language-learning game created using Unity, made within 36 hours for UofTHacks13.',
            tags: ['C#', 'Gemini API', 'ElevenLabs', 'Hackathon', 'Unity', 'Team Project'],
        },
        {
            id: 'project7',
            href: 'https://github.com/emtato/girlmath',
            image: '/images/girlmathimg.png',
            title: 'Girl Math',
            description: 'An AI-powered journaling and reflection app that helps girls build confidence in STEM, made for Ellehacks2026',
            tags: ['Python', 'MongoDB', 'Gemini API', 'FastAPI', 'Hackathon', 'Team Project'],
        },
    ]

    return (
        <section id="projects-section">
            <div style={{ height: '4rem' }}></div>
            <h2 className="header-2">projects</h2>
            <p className="body">here are some recent projects I have worked on<br/></p>
            <div id="projects-container">
                {projects.map((project) => (
                    <div key={project.id} id={project.id} className="project">
                        <a
                            href={project.href}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="project-link"
                        >
                            <Image
                                src={project.image}
                                alt={project.title}
                                width={256}
                                height={256}
                                style={{ width: '100%', height: 'auto' }}
                            />
                            <div className="project-content">
                                <p className="body-smaller-bold">{project.title}</p>
                                <p className="body-smaller">{project.description}</p>
                                {project.tags && (
                                    <div className="project-tags">
                                        {project.tags.map((tag) => (
                                            <span key={tag} className="project-tag">{tag}</span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </a>
                    </div>
                ))}
            </div>
            <div className="row-flex">
                <p className="body">see more at:</p>
                <a href="https://github.com/amansdali" target="_blank" rel="noopener noreferrer">
                    <p className="link-text">github.com/amansdali</p>
                </a>
            </div>
            <div className="small-arrow-divider">
                <a href="#contact-section" aria-label="Navigate to contact section">
                    <svg className="small-arrow" viewBox="0 0 57 54" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <line x1="4.24264" y1="4" x2="29" y2="28.7574" strokeWidth="6" strokeLinecap="round"/>
                        <line x1="3" y1="-3" x2="38.0122" y2="-3" transform="matrix(0.707107 -0.707107 -0.707107 -0.707107 25 29)" strokeWidth="6" strokeLinecap="round"/>
                        <line x1="4.24264" y1="25" x2="29" y2="49.7574" strokeWidth="6" strokeLinecap="round"/>
                        <line x1="3" y1="-3" x2="38.0122" y2="-3" transform="matrix(0.707107 -0.707107 -0.707107 -0.707107 25 50)" strokeWidth="6" strokeLinecap="round"/>
                    </svg>
                </a>
            </div>
        </section>
    )
}
