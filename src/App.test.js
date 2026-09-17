import { fireEvent, render, screen } from '@testing-library/react';
import App from './App';

test('renders the LexBrief home page', () => {
  render(<App />);
  expect(screen.getAllByText('LEXBRIEF')).toHaveLength(2);
  expect(screen.getByText(/One case/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { pressed: true })).toHaveTextContent('Lawyer');
  expect(screen.getByRole('heading', { name: /Upload the lawyer case file/i })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: /Judge.*Precedents/i }));
  expect(screen.getByRole('heading', { name: /Upload the case file/i })).toBeInTheDocument();
  expect(screen.queryByRole('heading', { name: /Upload the lawyer case file/i })).not.toBeInTheDocument();
});
