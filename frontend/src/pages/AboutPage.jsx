import './AboutPage.css';
import teamPhoto from '../assets/team-photo.jpg';
import systemDiagram from '../assets/system-diagram.png';
import nepaliPlateExample from '../assets/nepali-plate.png';

export default function AboutPage() {
  return (
    <div className="about-container">
      <div className="about-hero">
        <h1>About Nepali License Plate Recognition</h1>
        <p className="hero-subtitle">Revolutionizing vehicle identification with Devanagari script recognition</p>
      </div>
      
      <div className="about-content">
        <section className="project-overview">
          <div className="text-content">
            <h2>Project Overview</h2>
            <p>
              Automatic Nepali License Plate Recognition (NLPR) system is specifically designed to recognize 
              Nepali license plates written in Devanagari script. Unlike existing systems that primarily work with 
              Latin characters, our solution addresses the unique challenges of Nepali script recognition.
            </p>
            <p>
              Our system aims to address the gap in existing recognition systems
which largely overlooks non-English scripts and are not optimized for Nepal., making it suitable for real-world applications in traffic 
              management, parking systems, and law enforcement.
            </p>
          </div>
          <div className="image-content">
            <img src={nepaliPlateExample} alt="Example of Nepali license plate" />
            <p className="image-caption">Example of license plate</p>
          </div>
        </section>

        <section className="problem-statement">
          <h2>The Challenge</h2>
          <div className="problem-points">
            <div className="problem-card">
              <h3>Script Complexity</h3>
              <p>Devanagari script has complex character shapes and modifiers that challenge traditional OCR systems.</p>
            </div>
            <div className="problem-card">
              <h3>Environmental Factors</h3>
              <p>Real-world conditions like poor lighting, motion blur, and weather affect recognition accuracy.</p>
            </div>
            <div className="problem-card">
              <h3>Limited Research</h3>
              <p>Most existing systems are designed for Latin scripts, leaving a gap for Nepali applications.</p>
            </div>
          </div>
        </section>

        <section className="technology-stack">
          <h2>Our Innovative Solution</h2>
          <div className="tech-details">
            <div className="system-diagram">
              <img src={systemDiagram} alt="System architecture diagram" />
            </div>
            <div className="tech-list">
              <h3>Technology Stack</h3>
              <ul>
                <li>
                  <strong>YOLOv8</strong> - For accurate license plate detection with bounding box precision
                </li>
                <li>
                  <strong>CRNN (CNN + LSTM)</strong> - Combines convolutional and recurrent networks for sequence recognition
                </li>
                <li>
                  <strong>CTC Loss</strong> - Handles alignment between input images and output sequences
                </li>
                <li>
                  <strong>React.js</strong> - Modern frontend for intuitive user experience
                </li>
                <li>
                  <strong>FastAPI</strong> - Efficient backend service for model inference
                </li>
              </ul>
            </div>
          </div>
        </section>

        <section className="key-features">
          <h2>Key Features</h2>
          <div className="features-grid">
            <div className="feature-card">
              
              <h3>Vehicle Detection</h3>
              <p>Identifies vehicles in images 
                and process it for recognizing license plate</p>
            </div>
            <div className="feature-card">
              <h3>Plate Localization</h3>
              <p>Locates license plate regions within vehicle images</p>
            </div>
            <div className="feature-card">
              <h3>Devanagari OCR</h3>
              <p>Specialized recognition for Nepali script characters</p>
            </div>
            <div className="feature-card">
              <h3>Recognition+Processing</h3>
              <p>Optimized for quick analysis suitable image applications</p>
            </div>
          </div>
        </section>

        <section className="team-section">
          <h2>Our Team</h2>
          <div className="team-content">
            <div className="team-photo">
              <img src={teamPhoto} alt="Team 200_OK working on the project" />
            </div>
            <div className="team-description">
              <p>
                We are Team 200_OK from Nepal College of Information Technology, passionate about 
                solving real-world problems through innovative technology solutions. Our diverse 
                skills in computer vision, machine learning, and software development come 
                together to create this specialized system for Nepal's needs.
              </p>
              <p>
                This project represents our academic research in computer vision and machine 
                learning, with the goal of contributing to smarter transportation solutions in Nepal.
              </p>
            </div>
          </div>
        </section>

        <section className="applications">
          <h2>Potential Applications</h2>
          <div className="applications-list">
            <div className="application-card">
              <h3>Traffic Monitoring</h3>
              <p>Automated vehicle identification for traffic management systems</p>
            </div>
            <div className="application-card">
              <h3>Parking Systems</h3>
              <p>Streamlined entry/exit management for parking facilities</p>
            </div>
            <div className="application-card">
              <h3>Law Enforcement</h3>
              <p>Support for identifying vehicles of interest</p>
            </div>
            <div className="application-card">
              <h3>Smart Cities</h3>
              <p>Integration with intelligent transportation infrastructure</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}