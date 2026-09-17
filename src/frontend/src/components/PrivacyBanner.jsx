import "./PrivacyBanner.css";

export default function PrivacyBanner({ text }) {
  return (
    <div className="privacy-banner">
      <span className="privacy-banner-icon">&#128274;</span>
      <span>{text}</span>
    </div>
  );
}
